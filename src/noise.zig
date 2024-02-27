const std = @import("std");

const f64x4 = @Vector(4, f64);

pub const Result = struct {
    val: f64,
    grad: @Vector(3, f64),
};

pub const Simplex = struct {
    seed: u64,

    pub fn get(self: Simplex, x: f64, y: f64, z: f64) Result {
        return simplex(self.seed, x, y, z);
    }
};

pub const Fractal = struct {
    seed: u64,
    octaves: usize,
    persistence: f64,
    lacunarity: f64,

    pub fn get(self: Fractal, x: f64, y: f64, z: f64) Result {
        std.debug.assert(self.octaves > 0);
        std.debug.assert(0.0 <= self.persistence and self.persistence <= 1.0);
        std.debug.assert(self.lacunarity >= 1.0);
        var val: f64 = 0.0;
        var grad: @Vector(3, f64) = @splat(0);
        var seed = self.seed;
        var freq: f64 = 1.0;
        var ampl: f64 = 1.0;
        var max_ampl: f64 = 0.0;

        for (0..self.octaves) |_| {
            const result = simplex(seed, x * freq, y * freq, z * freq);
            val += result.val * ampl;
            grad += result.grad * @as(@Vector(3, f64), @splat(freq * ampl));
            max_ampl += ampl;
            freq *= self.lacunarity;
            ampl *= self.persistence;
            seed *%= 0x7FF13E69C31A371B;
        }

        return .{
            .val = val / max_ampl,
            .grad = grad / @as(@Vector(3, f64), @splat(max_ampl)),
        };
    }
};

// based on https://github.com/stegu/psrdnoise/, by Stefan Gustavson and Ian MacEwan
// ported to Zig and modified by Akshay Trivedi
fn simplex(seed: u64, x: f64, y: f64, z: f64) Result {
    const px: f64x4 = @splat(x);
    const py: f64x4 = @splat(y);
    const pz: f64x4 = @splat(z);

    // skew into simplectic coordinates
    const skew: f64x4 = @splat((x + y + z) * (1.0 / 3.0));
    const uvw = f64x4{ x, y, z, 0.0 } + skew;

    // get skew coordinates for base lattice vertex & inner skew coordinates
    const j0 = @floor(uvw);
    const f0 = uvw - j0;

    // determine remaining lattice vertices using inner skew coordinates
    const o1, const o2 = blk: {
        const cmp_lhs = @shuffle(f64, f0, undefined, [_]isize{ 0, 1, 0, 3 });
        const cmp_rhs = @shuffle(f64, f0, undefined, [_]isize{ 1, 2, 2, 3 });
        const lt = @select(f64, cmp_lhs < cmp_rhs, @as(f64x4, @splat(1.0)), @as(f64x4, @splat(0.0)));
        const gt = @as(f64x4, @splat(1.0)) - lt;
        const cmp1 = @shuffle(f64, gt, lt, [_]isize{ 2, -1, -2, -4 });
        const cmp2 = @shuffle(f64, gt, lt, [_]isize{ 0, 1, -3, 3 });
        break :blk .{ @min(cmp1, cmp2), @max(cmp1, cmp2) };
    };

    // get remaining lattice skew coordinates
    const jx = @as(f64x4, @splat(j0[0])) + @shuffle(f64, o1, o2, [_]isize{ 3, 0, -1, -4 });
    const jy = @as(f64x4, @splat(j0[1])) + @shuffle(f64, o1, o2, [_]isize{ 3, 1, -2, -4 });
    const jz = @as(f64x4, @splat(j0[2])) + @shuffle(f64, o1, o2, [_]isize{ 3, 2, -3, -4 });

    // return lattice vertices to real coordinates
    const unskew = (jx + jy + jz) * @as(f64x4, @splat(1.0 / 6.0));
    const dx = px - jx + unskew;
    const dy = py - jy + unskew;
    const dz = pz - jz + unskew;

    // generate gradients
    const gx, const gy, const gz = randomGradients(seed, jx, jy, jz);
    const gdotx = gx * dx + gy * dy + gz * dz;

    // determine contribution of each lattice vertex
    const w = @max(@as(f64x4, @splat(0.5)) - (dx * dx + dy * dy + dz * dz), @as(f64x4, @splat(0.0)));
    const sw2 = w * w * @as(f64x4, @splat(8192.0 / 125.0 * @sqrt(1.0 / 3.0)));
    const sw3 = sw2 * w;
    const dsw = sw2 * gdotx * @as(f64x4, @splat(-6.0));

    // calculate noise & gradient
    const val = @reduce(.Add, sw3 * gdotx);
    const grad = .{
        @reduce(.Add, sw3 * gx + dsw * dx),
        @reduce(.Add, sw3 * gy + dsw * dy),
        @reduce(.Add, sw3 * gz + dsw * dz),
    };

    return .{ .val = val, .grad = grad };
}

fn randomGradients(seed: u64, px: f64x4, py: f64x4, pz: f64x4) [3]f64x4 {
    const u32x4 = @Vector(4, u32);
    const u64x4 = @Vector(4, u64);
    const PRIME64_1: u64x4 = @splat(0x9E3779B185EBCA87);
    const PRIME64_2: u64x4 = @splat(0xC2B2AE3D27D4EB4F);
    const PRIME64_3: u64x4 = @splat(0x165667B19E3779F9);
    const PRIME64_4: u64x4 = @splat(0x85EBCA77C2B2AE63);
    const PRIME64_5: u64x4 = @splat(0x27D4EB2F165667C5);

    // preform a simplified XxHash64 algorithm.
    // implement SIMD over the 4 points, rather than over lanes.
    // each XxHash64 input is 24 bytes - 3 (8 byte) floats: x, y, z
    var acc: u64x4 = @splat(seed);
    acc +%= PRIME64_5;
    acc +%= @as(u64x4, @splat(24)); // inputLength
    inline for (.{ px, py, pz }) |p| {
        const lane: u64x4 = @bitCast(p);
        // acc = acc xor round(0, lane);
        acc ^= std.math.rotl(u64x4, lane *% PRIME64_2, 31) *% PRIME64_1;
        // acc = (acc <<< 27) * PRIME64_1 + PRIME64_4;
        acc = std.math.rotl(u64x4, acc, 27) *% PRIME64_1 +% PRIME64_4;
    }
    // avalanche
    acc ^= std.math.shr(u64x4, acc, 33);
    acc *%= PRIME64_2;
    acc ^= std.math.shr(u64x4, acc, 29);
    acc *%= PRIME64_3;
    acc ^= std.math.shr(u64x4, acc, 32);

    // extract hash
    const acc0: u32x4 = @truncate(acc);
    const acc1: u32x4 = @truncate(std.math.shr(u64x4, acc, 32));
    const acc0_f64 = f64x4{ @floatFromInt(acc0[0]), @floatFromInt(acc0[1]), @floatFromInt(acc0[2]), @floatFromInt(acc0[3]) };
    const acc1_f64 = f64x4{ @floatFromInt(acc1[0]), @floatFromInt(acc1[1]), @floatFromInt(acc1[2]), @floatFromInt(acc1[3]) };

    // parameters: y (altitude), t (azimuth)
    // y: (-1, 1)
    const y = acc0_f64 * @as(f64x4, @splat(2.0 / 4294967296.0)) + @as(f64x4, @splat(-4294967295.0 / 4294967296.0));
    // t: [0, 2pi)
    const t = acc1_f64 * @as(f64x4, @splat(std.math.tau / 4294967296.0));

    // generate gradients
    const r = @sqrt(@as(f64x4, @splat(1.0)) - y * y);
    return .{ r * @cos(t), y, r * @sin(t) };
}

const test_cases = struct {
    const noises = .{
        Simplex{ .seed = 0 },
        Simplex{ .seed = 0xFFFFFFFFFFFFFFFF },
        Simplex{ .seed = 0x1337 },
        Simplex{ .seed = 0x9B080480ADA8697F },
        Fractal{ .seed = 1337, .octaves = 3, .persistence = 0.5, .lacunarity = 2.0 },
    };
    const points = [_][3]f64{
        .{ 0, 0, 0 },
        .{ 1, 1, 1 },
        .{ 1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0 },
        .{ 13.37, -3.1415, 2.71828 },
        .{ -1.4142, 1.618, 0.69314 },
    };

    const test_range = 100000;
    const n_rand_points = 100000;

    fn run(
        comptime f: fn (anytype, f64, f64, f64) anyerror!void,
    ) anyerror!void {
        var prng = std.rand.DefaultPrng.init(0);
        const rand = prng.random();

        inline for (noises) |n| {
            for (points) |point| {
                try f(n, point[0], point[1], point[2]);
            }

            for (0..n_rand_points) |_| {
                const x = rand.floatNorm(f64);
                const y = rand.floatNorm(f64);
                const z = rand.floatNorm(f64);
                try f(n, x, y, z);
                try f(n, x * test_range, y * test_range, z * test_range);
            }
        }
    }
};

test "gradient" {
    const ds = 1e-6;
    const tol = 1e-9;

    try test_cases.run(struct {
        fn f(n: anytype, x: f64, y: f64, z: f64) anyerror!void {
            const r1 = n.get(x, y, z);

            // test the gradient on a cube of side length ds
            for (1..8) |i| {
                const xb = i & 1 != 0;
                const yb = i & 2 != 0;
                const zb = i & 4 != 0;

                const r2 = n.get(
                    if (xb) x + ds else x,
                    if (yb) y + ds else y,
                    if (zb) z + ds else z,
                );

                // approximation of r2.val - r1.val using gradient
                var del: f64 = 0.0;
                if (xb) del += r1.grad[0];
                if (yb) del += r1.grad[1];
                if (zb) del += r1.grad[2];
                del *= ds;

                try std.testing.expectApproxEqAbs(r2.val - r1.val, del, tol);
            }
        }
    }.f);
}

test "range" {
    try test_cases.run(struct {
        fn f(n: anytype, x: f64, y: f64, z: f64) anyerror!void {
            const r1 = n.get(x, y, z);
            try std.testing.expect(@abs(r1.val) <= 1);
        }
    }.f);
}

const std = @import("std");

pub fn add(lhs: [3]f64, rhs: [3]f64) [3]f64 {
    return .{ lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2] };
}

pub fn sub(lhs: [3]f64, rhs: [3]f64) [3]f64 {
    return .{ lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2] };
}

pub fn dot(lhs: [3]f64, rhs: [3]f64) f64 {
    return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

pub fn mag(vec: [3]f64) f64 {
    return @sqrt(dot(vec, vec));
}

pub fn cross(lhs: [3]f64, rhs: [3]f64) [3]f64 {
    return .{
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    };
}

pub fn normalize(vec: [3]f64) [3]f64 {
    const m = mag(vec);
    return .{ vec[0] / m, vec[1] / m, vec[2] / m };
}

pub fn slerp(from: [3]f64, to: [3]f64, t: f64) [3]f64 {
    const omega = std.math.acos(dot(from, to));
    const sin_omega = @sin(omega);
    const from_scale = @sin((1 - t) * omega) / sin_omega;
    const to_scale = @sin(t * omega) / sin_omega;

    return .{
        from_scale * from[0] + to_scale * to[0],
        from_scale * from[1] + to_scale * to[1],
        from_scale * from[2] + to_scale * to[2],
    };
}

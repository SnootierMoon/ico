const std = @import("std");
const lm = @import("lm.zig");
const SimplexNoise = @import("simplex.zig").Noise;

const Icosphere = @This();

n: usize,

verts: [][3]f64,
faces: [][3]usize,

pub fn init(allocator: std.mem.Allocator, seed: u32, subdivisions: usize) !Icosphere {
    const n = subdivisions + 1;
    const num_verts = 10 * n * n + 2;
    const num_faces = 20 * n * n;

    var self = Icosphere{
        .n = n,
        .verts = try allocator.alloc([3]f64, num_verts),
        .faces = try allocator.alloc([3]usize, num_faces),
    };

    for (0..12) |vert_idx| {
        self.verts[vert_idx] = base.verts[vert_idx];
    }

    for (0..30) |edge_idx| {
        self.generateEdge(edge_idx);
    }

    for (0..20) |face_idx| {
        self.generateFace(face_idx);
    }

    const noise = SimplexNoise{ .seed = seed };
    for (self.verts) |*vert| {
        _ = noise;
        const s = 1.0; // noise.get(vert[0], vert[1], vert[2]).val;
        vert[0] *= s;
        vert[1] *= s;
        vert[2] *= s;
    }

    return self;
}

pub fn deinit(self: Icosphere, allocator: std.mem.Allocator) void {
    allocator.free(self.verts);
    allocator.free(self.faces);
}

pub fn faceArea(self: Icosphere, face_idx: usize) f64 {
    const v0 = self.verts[self.faces[face_idx][0]];
    const v1 = self.verts[self.faces[face_idx][1]];
    const v2 = self.verts[self.faces[face_idx][2]];
    return lm.mag(lm.cross(lm.sub(v1, v0), lm.sub(v2, v0))) / 2;
}

fn generateEdge(self: *Icosphere, edge_idx: usize) void {
    const base_vert_idx = edge_idx * (self.n - 1) + 12;
    const v0 = base.verts[base.edges[edge_idx][0]];
    const v1 = base.verts[base.edges[edge_idx][1]];
    for (self.verts[base_vert_idx .. base_vert_idx + self.n - 1], 0..self.n - 1) |*v, i| {
        const t = @as(f64, @floatFromInt((i + 1))) / @as(f64, @floatFromInt(self.n));
        v.* = lm.slerp(v0, v1, t);
    }
}

fn generateFace(self: *Icosphere, face_idx: usize) void {
    if (self.n == 1) {
        self.faces[face_idx] = base.faces[face_idx];
    } else {
        const face = FaceInfo.init(self.n, face_idx);

        var curr_vert_idx = face.base_vert_idx;
        for (2..self.n) |i| {
            for (1..i) |j| {
                const cpQ_idx = face.eAB.query(self.n, false, i - j);
                const cpR_idx = face.eAB.query(self.n, false, i);
                const cpS_idx = face.eBC.query(self.n, false, j);
                const cpT_idx = face.eBC.query(self.n, true, i - j);
                const cpU_idx = face.eCA.query(self.n, true, i);
                const cpV_idx = face.eCA.query(self.n, true, j);

                const tQT = @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(self.n - (i - j)));
                const tRU = @as(f64, @floatFromInt(j)) / @as(f64, @floatFromInt(i));
                const tVS = @as(f64, @floatFromInt(i - j)) / @as(f64, @floatFromInt(self.n - j));

                const cpQT = lm.slerp(self.verts[cpQ_idx], self.verts[cpT_idx], tQT);
                const cpRU = lm.slerp(self.verts[cpR_idx], self.verts[cpU_idx], tRU);
                const cpVS = lm.slerp(self.verts[cpV_idx], self.verts[cpS_idx], tVS);

                const sum = lm.add(cpQT, lm.add(cpRU, cpVS));

                self.verts[curr_vert_idx] = lm.normalize(sum);
                curr_vert_idx += 1;
            }
        }

        var curr_face_idx = face.base_face_idx;
        for (0..self.n) |i| {
            self.faces[curr_face_idx] = .{
                face.query(self.n, i, 0),
                face.query(self.n, i + 1, 0),
                face.query(self.n, i + 1, 1),
            };
            curr_face_idx += 1;
            for (0..i) |j| {
                self.faces[curr_face_idx] = .{
                    face.query(self.n, i, j),
                    face.query(self.n, i + 1, j + 1),
                    face.query(self.n, i, j + 1),
                };
                self.faces[curr_face_idx + 1] = .{
                    face.query(self.n, i, j + 1),
                    face.query(self.n, i + 1, j + 1),
                    face.query(self.n, i + 1, j + 2),
                };
                curr_face_idx += 2;
            }
        }
    }
}

const FaceInfo = struct {
    vA_idx: usize,
    vB_idx: usize,
    vC_idx: usize,
    eAB: EdgeInfo,
    eBC: EdgeInfo,
    eCA: EdgeInfo,
    base_vert_idx: usize,
    base_face_idx: usize,

    fn init(n: usize, face_idx: usize) FaceInfo {
        const vA_idx = base.faces[face_idx][0];
        const vB_idx = base.faces[face_idx][1];
        const vC_idx = base.faces[face_idx][2];

        return FaceInfo{
            .vA_idx = vA_idx,
            .vB_idx = vB_idx,
            .vC_idx = vC_idx,
            .eAB = EdgeInfo.init(n, vA_idx, vB_idx),
            .eBC = EdgeInfo.init(n, vB_idx, vC_idx),
            .eCA = EdgeInfo.init(n, vC_idx, vA_idx),
            .base_vert_idx = face_idx * (n - 1) * (n - 2) / 2 + 30 * (n - 1) + 12,
            .base_face_idx = face_idx * n * n,
        };
    }

    fn query(face: FaceInfo, n: usize, i: usize, j: usize) usize {
        return if (i == 0 and j == 0)
            face.vA_idx
        else if (i == n and j == 0)
            face.vB_idx
        else if (i == n and j == n)
            face.vC_idx
        else if (j == 0)
            face.eAB.query(n, false, i)
        else if (i == n)
            face.eBC.query(n, false, j)
        else if (i == j)
            face.eCA.query(n, true, i)
        else
            face.base_vert_idx + (i - 2) * (i - 1) / 2 + j - 1;
    }
};

const EdgeInfo = struct {
    base_vert_idx: usize,
    rev: bool,

    fn init(n: usize, vA_idx: usize, vB_idx: usize) EdgeInfo {
        const edge: struct { idx: usize, rev: bool } = for (base.edges, 0..) |edge, idx| {
            if (edge[0] == vA_idx and edge[1] == vB_idx) {
                break .{ .idx = idx, .rev = false };
            } else if (edge[0] == vB_idx and edge[1] == vA_idx) {
                break .{ .idx = idx, .rev = true };
            }
        } else unreachable;

        return .{
            .base_vert_idx = edge.idx * (n - 1) + 12,
            .rev = edge.rev,
        };
    }

    fn query(edge: EdgeInfo, n: usize, rev: bool, idx: usize) usize {
        return if (edge.rev == rev)
            edge.base_vert_idx + idx - 1
        else
            edge.base_vert_idx + n - idx - 1;
    }
};

const base = struct {
    /// as = sqrt(0.5 - sqrt(0.05))
    const as = 0.525731112119133606025669084847876607285497932243341781528936;
    /// al = sqrt(0.5 + sqrt(0.05))
    const al = 0.850650808352039932181540497063011072240401403764816881836740;

    const verts = [12][3]f64{
        .{ al, as, 0.0 }, .{ al, -as, 0.0 }, .{ -al, as, 0.0 }, .{ -al, -as, 0.0 },
        .{ 0.0, al, as }, .{ 0.0, al, -as }, .{ 0.0, -al, as }, .{ 0.0, -al, -as },
        .{ as, 0.0, al }, .{ -as, 0.0, al }, .{ as, 0.0, -al }, .{ -as, 0.0, -al },
    };

    const faces = [20][3]usize{
        .{ 0, 1, 10 }, .{ 1, 0, 8 },  .{ 2, 3, 9 },   .{ 3, 2, 11 },
        .{ 4, 5, 2 },  .{ 5, 4, 0 },  .{ 6, 7, 1 },   .{ 7, 6, 3 },
        .{ 8, 9, 6 },  .{ 9, 8, 4 },  .{ 10, 11, 5 }, .{ 11, 10, 7 },
        .{ 0, 4, 8 },  .{ 0, 10, 5 }, .{ 1, 8, 6 },   .{ 1, 7, 10 },
        .{ 2, 9, 4 },  .{ 2, 5, 11 }, .{ 3, 6, 9 },   .{ 3, 11, 7 },
    };

    const edges = [30][2]usize{
        .{ 0, 1 },   .{ 0, 8 },  .{ 1, 8 },  .{ 0, 10 }, .{ 1, 10 },
        .{ 2, 3 },   .{ 2, 9 },  .{ 3, 9 },  .{ 2, 11 }, .{ 3, 11 },
        .{ 4, 5 },   .{ 4, 0 },  .{ 5, 0 },  .{ 4, 2 },  .{ 5, 2 },
        .{ 6, 7 },   .{ 6, 1 },  .{ 7, 1 },  .{ 6, 3 },  .{ 7, 3 },
        .{ 8, 9 },   .{ 8, 4 },  .{ 9, 4 },  .{ 8, 6 },  .{ 9, 6 },
        .{ 10, 11 }, .{ 10, 5 }, .{ 11, 5 }, .{ 10, 7 }, .{ 11, 7 },
    };
};

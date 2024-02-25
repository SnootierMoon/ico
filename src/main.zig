const std = @import("std");
const Icosphere = @import("Icosphere.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var opts = struct { subdivisions: usize = 10, output_path: []const u8 = "icosphere.obj" }{};
    if (args.len >= 2) {
        opts.subdivisions = try std.fmt.parseInt(usize, args[1], 10);
    }
    if (args.len >= 3) {
        opts.output_path = args[2];
    }

    const icosphere = try Icosphere.init(allocator, 1337, opts.subdivisions);
    defer icosphere.deinit(allocator);

    var sum: f64 = 0.0;
    var min = std.math.inf(f64);
    var max = -std.math.inf(f64);
    for (0..icosphere.faces.len) |face_idx| {
        const area = icosphere.faceArea(face_idx);
        sum += area;
        if (area < min) {
            min = area;
        }
        if (area > max) {
            max = area;
        }
    }
    const avg = sum / @as(f64, @floatFromInt(icosphere.faces.len));

    var square_error_sum: f64 = 0.0;
    for (0..icosphere.faces.len) |face_idx| {
        const err = (icosphere.faceArea(face_idx) - avg);
        square_error_sum += err * err;
    }
    const stdev = @sqrt(square_error_sum / @as(f64, @floatFromInt(icosphere.faces.len)));

    std.debug.print("Subdivisions: {}\n", .{opts.subdivisions});
    std.debug.print("Writing output to: {s}\n\n", .{opts.output_path});

    std.debug.print("Face Area Stats:\nMin: {}\nMax: {}\nRange: {}\n\n", .{
        min,
        max,
        max - min,
    });
    std.debug.print("Avg: {}\nStdev: {}\nRel Stdev: {}\n", .{
        avg,
        stdev,
        stdev / avg,
    });

    try renderObj(opts.output_path, icosphere.verts, icosphere.faces);
}

fn renderObj(filename: []const u8, verts: []const [3]f64, faces: []const [3]usize) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var bw = std.io.bufferedWriter(file.writer());
    const writer = bw.writer();

    for (verts) |vert| {
        try writer.print("v {d:.4} {d:.4} {d:.4}\n", .{ vert[0], vert[1], vert[2] });
    }

    for (faces) |face| {
        try writer.print("f {} {} {}\n", .{ face[0] + 1, face[1] + 1, face[2] + 1 });
    }

    try bw.flush();
}

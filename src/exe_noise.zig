const std = @import("std");
const FractalNoise = @import("noise.zig").Fractal;

fn writeBmp(filename: []const u8, width: u32, height: u32, seed: u64, x: f64, y: f64, scale: f64) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();
    var bw = std.io.bufferedWriter(file.writer());
    const writer = bw.writer();
    std.debug.print("Writing output to {s}\n", .{filename});

    const n = FractalNoise{ .seed = seed, .octaves = 5, .persistence = 0.5, .lacunarity = 2.0 };
    // bmp header
    try writer.writeAll("BM");
    try writer.writeInt(u32, 54 + 256 * 4 + width * height, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u16, 0, .little);
    try writer.writeInt(u32, 54 + 256 * 4, .little);

    // bmp info header
    try writer.writeInt(u32, 40, .little);
    try writer.writeInt(u32, width, .little);
    try writer.writeInt(u32, height, .little);
    try writer.writeInt(u16, 1, .little);
    try writer.writeInt(u16, 8, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, width * height, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, 0, .little);
    try writer.writeInt(u32, 256, .little);
    try writer.writeInt(u32, 256, .little);

    // palette
    for (0..256) |i| {
        try writer.writeByte(@intCast(i));
        try writer.writeByte(@intCast(i));
        try writer.writeByte(@intCast(i));
        try writer.writeByte(0);
    }

    for (0..height) |i| {
        for (0..width) |j| {
            const px = x + scale * (@as(f64, @floatFromInt(i + 1)) / @as(f64, @floatFromInt(width + 1)) - 0.5);
            const py = y + scale * (@as(f64, @floatFromInt(j + 1)) / @as(f64, @floatFromInt(width + 1)) - 0.5);
            const v: u8 = @intFromFloat(@min(255.0, 128 * (n.get(px, py, 0.0).val + 1)));
            try writer.writeByte(v);
        }
    }
}

pub fn main() !void {
    const seed: u64 = @truncate(@as(u128, @bitCast(std.time.nanoTimestamp())));
    std.debug.print("seed: {}\n", .{seed});

    try writeBmp("output.bmp", 640, 320, seed, 8.0, 2.1, 10.0);
}

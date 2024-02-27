const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe_ico = b.addExecutable(.{
        .name = "ico",
        .root_source_file = .{ .path = "src/exe_ico.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe_ico);

    const run_ico_cmd = b.addRunArtifact(exe_ico);
    run_ico_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_ico_cmd.addArgs(args);
    }
    const run_ico_step = b.step("run-ico", "run the app");
    run_ico_step.dependOn(&run_ico_cmd.step);

    const exe_noise = b.addExecutable(.{
        .name = "noise",
        .root_source_file = .{ .path = "src/exe_noise.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(exe_ico);

    const run_noise_cmd = b.addRunArtifact(exe_noise);
    run_noise_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_noise_cmd.addArgs(args);
    }
    const run_noise_step = b.step("run-noise", "Run the app");
    run_noise_step.dependOn(&run_noise_cmd.step);
}

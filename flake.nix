{
  description = "Dev shell for speech (Tauri + Rust + whisper.cpp)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            clang
            shaderc # provides glslc, needed to compile whisper.cpp's Vulkan shaders
          ];

          buildInputs = with pkgs; [
            openssl
            llvmPackages.libclang
            alsa-lib

            # Vulkan backend for whisper.cpp (offloads decode to the AMD iGPU)
            vulkan-loader
            vulkan-headers
            vulkan-tools # vulkaninfo, for verifying the GPU is visible

            # Tauri v2 Linux runtime deps
            gtk3
            webkitgtk_4_1
            libsoup_3
            glib
            cairo
            pango
            atk
            gdk-pixbuf
            librsvg
            dbus
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # Let the built binary find libvulkan at runtime inside the dev shell.
          # The RADV driver ICD itself is picked up from the host's
          # /run/opengl-driver, which nixpkgs' loader searches by default.
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:$LD_LIBRARY_PATH"
          '';
        };
      }
    );
}

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
        # tauri's AppImage bundler reads `pkg-config --libs-only-L ayatana-appindicator3-0.1`
        # as a single dir; the real .pc also emits every dependency's -L. Shadow it with one -L.
        trayPc = pkgs.writeTextDir "lib/pkgconfig/ayatana-appindicator3-0.1.pc" ''
          Name: ayatana-appindicator3-0.1
          Description: libayatana-appindicator path for tauri's AppImage bundler
          Version: ${pkgs.libayatana-appindicator.version}
          Libs: -L${pkgs.libayatana-appindicator}/lib -layatana-appindicator3
        '';
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
            libayatana-appindicator # tray icon (dlopen'd at runtime; AppImage bundler looks it up via pkg-config)

            # Screen replay (screen/stream.rs): PipeWire screencast via xdg-desktop-portal
            pipewire
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          # Let the built binary find libvulkan at runtime inside the dev shell.
          # The RADV driver ICD itself is picked up from the host's
          # /run/opengl-driver, which nixpkgs' loader searches by default.
          shellHook = ''
            export PKG_CONFIG_PATH="${trayPc}/lib/pkgconfig:$PKG_CONFIG_PATH"
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:${pkgs.libayatana-appindicator}/lib:$LD_LIBRARY_PATH"
            # GTK's file chooser (<input type="file">) aborts without its GSettings schemas.
            export XDG_DATA_DIRS="${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS"
          '';
        };
      }
    );
}

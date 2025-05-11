# This file is pretty general, and you can adapt it in your project replacing
# only `name` and `description` below.
{
  description = "NeuraTable - augmenting open source photography workflows with machine learning tools";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {inherit system;};
        #rust = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        runtimeDeps = [pkgs.vulkan-loader pkgs.openssl pkgs.libiconv pkgs.exiftool];
        buildDeps = [pkgs.pkg-config pkgs.makeWrapper];
      in {
        devShell = pkgs.mkShell {
          packages = [pkgs.rust.packages.stable.rustc pkgs.cargo pkgs.rustfmt pkgs.rust-analyzer pkgs.git-lfs] ++ runtimeDeps ++ buildDeps;
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.vulkan-loader];
        };

        packages.neuratable = let
          rustPlatform = pkgs.rust.packages.stable.rustPlatform;
        in
          rustPlatform.buildRustPackage {
            pname = "neuratable";
            version = "0.2.0";

            src = ./.;
            buildInputs = runtimeDeps;
            nativeBuildInputs = buildDeps;

            cargoHash = "sha256-8ryANOhOQUfLqfBU4ZhRPjM6yfRFmAWqgzKHDI3SRL8=";
            useFetchCargoVendor = true;

            postFixup = ''
              patchelf --add-rpath ${pkgs.vulkan-loader}/lib $out/bin/*
              for executable in $out/bin/*; do
                wrapProgram $executable --prefix PATH : ${pkgs.exiftool}/bin
              done
            '';

            meta = {
              description = "NeuraTable is a tool to augment open source photography workflows with machine learning tools";
              homepage = "https://github.com/mayjs/NeuraTable";
              license = pkgs.lib.licenses.gpl3Only;
              maintainers = [];
            };
          };

        packages.default = self.packages.${system}.neuratable;
      }
    );
}

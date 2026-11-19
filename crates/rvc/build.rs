fn main() {
    println!("cargo:rustc-check-cfg=cfg(rvc_cuda_sdpa)");
    #[cfg(feature = "index-training")]
    {
        println!("cargo:rerun-if-env-changed=FAISS_LIB_DIR");
        if let Some(directory) = std::env::var_os("FAISS_LIB_DIR") {
            let directory = std::path::PathBuf::from(directory);
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
    }
    #[cfg(feature = "inference")]
    build_cuda_sdpa();
    #[cfg(feature = "cuda-graph")]
    cuda_graph::build();
}

#[cfg(feature = "inference")]
fn build_cuda_sdpa() {
    use std::{env, path::PathBuf};

    const BRIDGE: &str = "src/separation/roformer/cuda_sdpa.cpp";
    println!("cargo:rerun-if-changed={BRIDGE}");
    for key in [
        "LIBTORCH",
        "LIBTORCH_INCLUDE",
        "LIBTORCH_LIB",
        "LIBTORCH_CXX11_ABI",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let target = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS");
    let library = match target.as_str() {
        "linux" => "libtorch_cuda.so",
        "windows" => "torch_cuda.lib",
        _ => return,
    };
    let root = env::var_os("LIBTORCH").map(PathBuf::from);
    let lib = env::var_os("LIBTORCH_LIB")
        .map(PathBuf::from)
        .or_else(|| root.clone())
        .map(|path| path.join("lib"));
    let Some(lib) = lib.filter(|path| path.join(library).is_file()) else {
        return;
    };
    let include = env::var_os("LIBTORCH_INCLUDE")
        .map(PathBuf::from)
        .or(root)
        .expect("set LIBTORCH_INCLUDE when linking a separate CUDA LibTorch library");
    let mut compiler = cc::Build::new();
    compiler
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include(include.join("include"))
        .file(BRIDGE);
    if target == "linux" {
        let abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".into());
        compiler.define("_GLIBCXX_USE_CXX11_ABI", Some(abi.as_str()));
    } else {
        compiler.flag_if_supported("/EHsc");
    }
    compiler.compile("rvc_cuda_sdpa");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-cfg=rvc_cuda_sdpa");
}

#[cfg(feature = "cuda-graph")]
mod cuda_graph {
    use std::env;
    use std::path::PathBuf;

    const BRIDGE: &str = "src/model/cuda_graph/bridge.cpp";

    pub fn build() {
        println!("cargo:rerun-if-changed={BRIDGE}");
        for name in [
            "LIBTORCH",
            "LIBTORCH_INCLUDE",
            "LIBTORCH_LIB",
            "LIBTORCH_CXX11_ABI",
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_INCLUDE_DIRS",
        ] {
            println!("cargo:rerun-if-env-changed={name}");
        }

        let target = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS");
        assert!(
            matches!(target.as_str(), "linux" | "windows"),
            "cuda-graph requires a Linux or Windows CUDA LibTorch build"
        );
        let libtorch = env::var_os("LIBTORCH")
            .map(PathBuf::from)
            .expect("set LIBTORCH to the unpacked CUDA-enabled LibTorch directory for cuda-graph");
        let include = env::var_os("LIBTORCH_INCLUDE").map_or_else(
            || libtorch.join("include"),
            |path| PathBuf::from(path).join("include"),
        );
        let lib = env::var_os("LIBTORCH_LIB").map_or_else(
            || libtorch.join("lib"),
            |path| PathBuf::from(path).join("lib"),
        );
        let lib_name = if target == "windows" {
            "torch_cuda.lib"
        } else {
            "libtorch_cuda.so"
        };
        assert!(
            lib.join(lib_name).is_file(),
            "cuda-graph requires CUDA-enabled LibTorch, missing {}",
            lib.join(lib_name).display()
        );
        assert!(
            include.join("ATen/cuda/CUDAGraph.h").is_file(),
            "missing CUDA Graph headers under {}",
            include.display()
        );

        let toolkit = env::var_os("CUDA_HOME")
            .or_else(|| env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .or_else(|| (target == "linux").then(|| PathBuf::from("/usr/local/cuda")));
        let mut cuda_includes: Vec<PathBuf> = env::var_os("CUDA_INCLUDE_DIRS")
            .iter()
            .flat_map(env::split_paths)
            .collect();
        if let Some(toolkit) = &toolkit {
            let include = toolkit.join("include");
            if include.is_dir() {
                cuda_includes.push(include);
            }
        }
        for header in ["cuda_runtime_api.h", "crt/host_defines.h"] {
            assert!(
                cuda_includes.iter().any(|path| path.join(header).is_file()),
                "missing {header}; set CUDA_HOME or CUDA_INCLUDE_DIRS to CUDA toolkit header directories"
            );
        }

        let mut compiler = cc::Build::new();
        compiler
            .cpp(true)
            .std("c++17")
            .warnings(false)
            .include(&include)
            .includes(&cuda_includes)
            .file(BRIDGE);
        if target == "linux" {
            let abi = env::var("LIBTORCH_CXX11_ABI").unwrap_or_else(|_| "1".into());
            compiler.define("_GLIBCXX_USE_CXX11_ABI", Some(abi.as_str()));
        } else {
            compiler.flag_if_supported("/EHsc");
        }

        compiler.compile("rvc_cuda_graph");
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=dylib=torch_cuda");
        println!("cargo:rustc-link-lib=dylib=c10_cuda");
        if target == "linux" {
            println!("cargo:rustc-link-lib=dl");
        }
    }
}

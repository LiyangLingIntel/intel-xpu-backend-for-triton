import contextlib
import sys
import io
import sysconfig
import os
import shutil
import subprocess
import setuptools


def is_xpu():
    import torch
    return torch.xpu.is_available()


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _cc_cmd(cc, src, out, include_dirs, library_dirs, libraries):
    if cc in ["cl", "clang-cl"]:
        cc_cmd = [cc, src, "/nologo", "/O2", "/LD"]
        cc_cmd += [f"/I{dir}" for dir in include_dirs]
        cc_cmd += [f"/Fo{os.path.join(os.path.dirname(out), 'main.obj')}"]
        cc_cmd += ["/link"]
        cc_cmd += [f"/OUT:{out}"]
        cc_cmd += [f"/IMPLIB:{os.path.join(os.path.dirname(out), 'main.lib')}"]
        cc_cmd += [f"/PDB:{os.path.join(os.path.dirname(out), 'main.pdb')}"]
        cc_cmd += [f"/LIBPATH:{dir}" for dir in library_dirs]
        cc_cmd += [f'{lib}.lib' for lib in libraries]
    else:
        cc_cmd = [cc, src, "-O3", "-shared", "-Wno-psabi"]
        if os.name != "nt":
            cc_cmd += ["-fPIC"]
        cc_cmd += [f'-l{lib}' for lib in libraries]
        cc_cmd += [f"-L{dir}" for dir in library_dirs]
        cc_cmd += [f"-I{dir}" for dir in include_dirs]
        cc_cmd += ["-o", out]

    return cc_cmd


def _build(name, src, srcdir, library_dirs, include_dirs, libraries, extra_compile_args=[]):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if os.name == "nt":
            cc = shutil.which("cl")
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]

    if is_xpu():
        icpx = None
        cxx = os.environ.get("CXX")
        if cxx is None:
            clangpp = shutil.which("clang++")
            gxx = shutil.which("g++")
            icpx = shutil.which("icpx")
            cxx = icpx if os.name == "nt" else icpx or clangpp or gxx
            if cxx is None:
                raise RuntimeError("Failed to find C++ compiler. Please specify via CXX environment variable.")
        cc = cxx
        import numpy as np
        numpy_include_dir = np.get_include()
        include_dirs = include_dirs + [numpy_include_dir]
        if icpx is not None:
            extra_compile_args += ["-fsycl"]
        else:
            extra_compile_args += ["--std=c++17"]
        if os.name == "nt":
            library_dirs = library_dirs + [os.path.join(sysconfig.get_paths(scheme=scheme)["stdlib"], "..", "libs")]
    else:
        cc_cmd = [cc]

    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = _cc_cmd(cc, src, so, include_dirs, library_dirs, libraries)
    cc_cmd += extra_compile_args

    if os.getenv("VERBOSE"):
        print(" ".join(cc_cmd))

    ret = subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    if ret == 0:
        return so
    # extra arguments
    extra_link_args = []
    # create extension module
    ext = setuptools.Extension(
        name=name,
        language='c',
        sources=[src],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args + ['-O3' if os.name != "nt" else "/O2"],
        extra_link_args=extra_link_args,
        library_dirs=library_dirs,
        libraries=libraries,
    )
    # build extension module
    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        script_args=args,
    )
    with quiet():
        setuptools.setup(**args)
    return so

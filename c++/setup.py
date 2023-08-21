from distutils.core import setup, Extension
import os

libfranka_include = os.path.expanduser('~/libfranka/include')
libfranka_lib = os.path.expanduser('/usr/local/lib/libfranka.so')

#all_macros = [('MOTION_DEBUG', None), ('PYTHON', None)]
all_macros = [('PYTHON', None)]
vo_macros = [] #[('VO_RESTRICT', None)]
so3_macros = [
        ('SO3_STRICT', None),
    ]

franka = Extension('franka_motion',
                    sources = ['franka_module.cpp'],
                    include_dirs = [libfranka_include],
                    extra_compile_args = ["-O3"],
                    define_macros=all_macros + vo_macros + so3_macros,
                    extra_objects=[libfranka_lib],
                    #extra_link_args=["-llibfranka.so"]
                    )

setup (name = 'franka_motion',
       version = '1.0',
       description = 'franka motion api',
       ext_modules = [franka])

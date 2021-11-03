Place the public header files in this directory. They will be
available to your code (and other modules) with

     #include <IMP/pynet/myheader.h>

All headers should include `IMP/pynet/pynet_config.h` as their
first include and surround all code with `IMPPYNET_BEGIN_NAMESPACE`
and `IMPPYNET_END_NAMESPACE` to put it in the IMP::pynet namespace
and manage compiler warnings.

Headers should also be exposed to SWIG in the `pyext/swig.i-in` file.

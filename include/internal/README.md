Place the private header files in this directory. They will be
available to your code with

     #include <IMP/pynet/internal/myheader.h>

All headers should include `IMP/pynet/pynet_config.h` as their
first include and surround all code with `IMPPYNET_BEGIN_INTERNAL_NAMESPACE`
and `IMPPYNET_END_INTERNAL_NAMESPACE` to put it in the
IMP::pynet::internal namespace and manage compiler warnings.

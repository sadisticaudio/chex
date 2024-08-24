#include "Chex.h"

BOOST_PYTHON_MODULE(chex)
{
    bp::class_<CheckpointProcessor>("CheckpointProcessor")//, bp::init<bp::dict>())
        .def("__init__", bp::make_constructor(makeCPProcessor))
        .def_readwrite("nam", &CheckpointProcessor::nam)
        .def("compare_cache", &CheckpointProcessor::compare_cache)
    ;
}
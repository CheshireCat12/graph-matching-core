from graph_pkg.utils.coordinator.coordinator cimport Coordinator
from graph_pkg.utils.coordinator.coordinator_classifier cimport CoordinatorClassifier


cpdef str _gr_name_to_df_name(str name)

cpdef str _gr_name_AIDS(str name, dict classes)

cpdef str _gr_name_mutagenicity(str filename, str name, dict classes)

cpdef dict _get_classes(CoordinatorClassifier coordinator)

cpdef void run_letter()

cpdef void run_AIDS()

cpdef void run_mutagenicity()

cpdef double[:, ::1] run(Coordinator coordinator)
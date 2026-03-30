from django.urls import path
from . import views

app_name = "markov"

urlpatterns = [
    # Pages
    path("", views.dashboard, name="dashboard"),
    path("results/", views.results_gallery, name="results_gallery"),
    path("gallery/", views.image_gallery, name="image_gallery"),
    path("analysis/", views.unified_analysis, name="unified_analysis"),
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
    path("compare/", views.compare_view, name="compare"),
    path("compare/sweep/", views.compare_sweep, name="compare_sweep"),
    path("compare/dem-markov/", views.compare_dem_markov, name="compare_dem_markov"),
    path("rsd/", views.rsd_analysis, name="rsd_analysis"),
    path("matrix/", views.matrix_analysis, name="matrix_analysis"),
    path("metrics/", views.metrics_analysis, name="metrics_analysis"),
    path("partitions/", views.partition_viewer, name="partition_viewer"),
    path(
        "partitions/compare/", views.partition_comparison, name="partition_comparison"
    ),
    path("partitioner-3d/", views.partitioner_3d, name="partitioner_3d"),
    path(
        "partitioner-3d/trame/", views.partitioner_3d_trame, name="partitioner_3d_trame"
    ),
    path(
        "api/partitions-pyvista/",
        views.api_partitions_pyvista,
        name="api_partitions_pyvista",
    ),
    # API
    path("api/matrix/<int:pk>/", views.api_matrix_data, name="api_matrix"),
    path("api/rsd/<int:pk>/", views.api_rsd_data, name="api_rsd"),
    path("api/stats/", views.api_experiment_stats, name="api_stats"),
    path("api/compare-rsd/", views.api_compare_rsd, name="api_compare_rsd"),
    path("api/partitions/", views.api_partitions, name="api_partitions"),
    path("api/dem-particles/", views.api_dem_particles, name="api_dem_particles"),
    path("api/comparison/", views.api_comparison, name="api_comparison"),
    # Unified Analysis Tab APIs
    path("api/tab/comparison/", views.api_comparison_tab, name="api_comparison_tab"),
    path("api/tab/rsd/", views.api_rsd_tab, name="api_rsd_tab"),
    path("api/tab/mixing/", views.api_mixing_tab, name="api_mixing_tab"),
    path("api/tab/eigenvalues/", views.api_eigenvalues_tab, name="api_eigenvalues_tab"),
    path(
        "api/tab/dem-markov/", views.api_dem_vs_markov_tab, name="api_dem_vs_markov_tab"
    ),
    # 3D Partitioner APIs
    path(
        "api/partitioner/schemas/",
        views.api_partitioner_schemas,
        name="api_partitioner_schemas",
    ),
    path(
        "api/partitioner/3d-data/",
        views.api_partitioner_3d_data,
        name="api_partitioner_3d_data",
    ),
    path(
        "api/partitioner/dem-markov-3d/",
        views.api_dem_vs_markov_3d,
        name="api_dem_vs_markov_3d",
    ),
    path(
        "api/partitioner/trame-data/",
        views.api_partitioner_trame_data,
        name="api_partitioner_trame_data",
    ),
    path(
        "api/partitioner/particles/",
        views.api_partitioner_particles,
        name="api_partitioner_particles",
    ),
    # Unified Analysis APIs
    path(
        "api/analysis/experiments/",
        views.api_analysis_experiments,
        name="api_analysis_experiments",
    ),
    path(
        "api/analysis/generate-images/",
        views.api_analysis_generate_images,
        name="api_analysis_generate_images",
    ),
    path("api/analysis/data/", views.api_analysis_data, name="api_analysis_data"),
    path("api/rsd-comparison/", views.api_rsd_comparison, name="api_rsd_comparison"),
]

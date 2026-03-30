from django.shortcuts import render


def trame_viz(request):
    """Render template containing Trame 3D visualization iframe"""
    return render(request, "trame_app/trame_viz.html")

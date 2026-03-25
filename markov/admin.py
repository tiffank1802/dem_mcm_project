from django.contrib import admin
from django import forms
from django.utils.html import format_html, mark_safe
from .models import PartitionMethod, Experiment, TransitionMatrix, RSDResult, DEMSnapshot
import json


# ═══════════════════════════════════════════════════════════════
# FORMULAIRES
# ═══════════════════════════════════════════════════════════════

class PartitionMethodForm(forms.ModelForm):
    """Formulaire pour créer/éditer une méthode de partitionnement."""
    
    # Affichage des paramètres par méthode
    nx = forms.IntegerField(required=False, label="Résolution X", initial=5)
    ny = forms.IntegerField(required=False, label="Résolution Y", initial=5)
    nz = forms.IntegerField(required=False, label="Résolution Z", initial=5)
    
    nr = forms.IntegerField(required=False, label="Résolution radiale", initial=5)
    ntheta = forms.IntegerField(required=False, label="Résolution angulaire", initial=5)
    radial_mode = forms.ChoiceField(
        choices=[("equal_dr", "Equal dr"), ("equal_area", "Equal area")],
        required=False, label="Mode radial"
    )
    
    n_cells = forms.IntegerField(required=False, label="Nb cellules (Voronoï)", initial=125)
    
    max_particles = forms.IntegerField(required=False, label="Particles max (Octree)", initial=50)
    max_depth = forms.IntegerField(required=False, label="Profondeur max (Octree)", initial=4)
    
    class Meta:
        model = PartitionMethod
        fields = ["name", "label", "description", "n_cells", "n_cells_visited", "population_mean", "population_std", "population_cv"]
    
    def clean(self):
        cleaned_data = super().clean()
        method = cleaned_data.get("name")
        
        # Construire les paramètres en fonction de la méthode
        params = {}
        if method == "cartesian":
            params = {
                "nx": cleaned_data.get("nx") or 5,
                "ny": cleaned_data.get("ny") or 5,
                "nz": cleaned_data.get("nz") or 5,
            }
        elif method == "cylindrical":
            params = {
                "nr": cleaned_data.get("nr") or 5,
                "ntheta": cleaned_data.get("ntheta") or 5,
                "nz": cleaned_data.get("nz") or 5,
                "radial_mode": cleaned_data.get("radial_mode") or "equal_dr",
            }
        elif method == "voronoi":
            params = {
                "n_cells": cleaned_data.get("n_cells") or 125,
            }
        elif method == "octree":
            params = {
                "max_particles": cleaned_data.get("max_particles") or 50,
                "max_depth": cleaned_data.get("max_depth") or 4,
            }
        
        cleaned_data["params"] = params
        return cleaned_data
    
    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.parameters = self.cleaned_data["params"]
        if commit:
            instance.save()
        return instance


class PartitionMethodAdmin(admin.ModelAdmin):
    form = PartitionMethodForm
    list_display = ["name_display", "label", "n_cells", "population_cv", "created_at"]
    list_filter = ["name", "created_at"]
    search_fields = ["label"]
    readonly_fields = ["created_at"]
    
    fieldsets = (
        ("Identification", {
            "fields": ["name", "label", "description"]
        }),
        ("Paramètres du Partitionnement", {
            "fields": ["nx", "ny", "nz", "nr", "ntheta", "radial_mode", "n_cells", "max_particles", "max_depth"],
            "description": "Remplissez selon la méthode sélectionnée"
        }),
        ("Métriques de Qualité", {
            "fields": ["n_cells_visited", "population_mean", "population_std", "population_cv"]
        }),
        ("Métadonnées", {
            "fields": ["created_at"],
            "classes": ["collapse"]
        }),
    )
    
    def name_display(self, obj):
        colors = {
            "cartesian": "#1f77b4",
            "cylindrical": "#ff7f0e",
            "voronoi": "#2ca02c",
            "quantile": "#d62728",
            "octree": "#9467bd",
            "physics": "#8c564b",
        }
        color = colors.get(obj.name, "#999")
        return format_html(
            '<span style="color: {}; font-weight: bold;">●</span> {}',
            color, obj.get_name_display()
        )
    name_display.short_description = "Méthode"


# ═══════════════════════════════════════════════════════════════
# EXPÉRIENCES
# ═══════════════════════════════════════════════════════════════

class ExperimentForm(forms.ModelForm):
    """Formulaire pour créer/éditer une expérience."""
    
    # Discrétisation temporelle
    nlt = forms.IntegerField(label="NLT", initial=50, help_text="Nombre de paires snapshot")
    step_size = forms.IntegerField(label="Step size", initial=1, help_text="Intervalle temporel")
    start_index = forms.IntegerField(label="Start index", initial=0, help_text="Index du premier fichier")
    
    class Meta:
        model = Experiment
        fields = ["folder_name", "partition_method", "nlt", "step_size", "start_index", "n_states"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["partition_method"].queryset = PartitionMethod.objects.all().order_by("name", "n_cells")


class ExperimentAdmin(admin.ModelAdmin):
    form = ExperimentForm
    list_display = ["folder_name_short", "method_badge", "n_states", "nlt", "step_size", "start_index", "status"]
    list_filter = ["partition_method__name", "created_at", "nlt"]
    search_fields = ["folder_name"]
    readonly_fields = ["created_at", "folder_name"]
    
    fieldsets = (
        ("Identification", {
            "fields": ["folder_name"]
        }),
        ("Partitionnement Spatial", {
            "fields": ["partition_method", "n_states"]
        }),
        ("Discrétisation Temporelle", {
            "fields": ["nlt", "step_size", "start_index"],
            "description": "Paramètres d'apprentissage de la matrice"
        }),
        ("Métadonnées", {
            "fields": ["created_at"],
            "classes": ["collapse"]
        }),
    )
    
    def folder_name_short(self, obj):
        return obj.folder_name[:50]
    folder_name_short.short_description = "Nom"
    
    def method_badge(self, obj):
        colors = {
            "cartesian": "#1f77b4",
            "cylindrical": "#ff7f0e",
            "voronoi": "#2ca02c",
            "quantile": "#d62728",
            "octree": "#9467bd",
        }
        color = colors.get(obj.partition_method.name, "#999")
        return format_html(
            '<span style="background: {}; color: white; padding: 3px 8px; border-radius: 4px; font-size: 11px;">{}</span>',
            color, obj.partition_method.name
        )
    method_badge.short_description = "Méthode"
    
    def status(self, obj):
        if obj.matrix:
            return mark_safe('<span style="color: green;">✓ Matrice OK</span>')
        return mark_safe('<span style="color: orange;">⚠ Pas de matrice</span>')
    status.short_description = "Statut"


# ═══════════════════════════════════════════════════════════════
# MATRICE DE TRANSITION
# ═══════════════════════════════════════════════════════════════

class TransitionMatrixAdmin(admin.ModelAdmin):
    list_display = ["experiment", "diag_mean_display", "spectral_gap_display", "fraction_visited"]
    list_filter = ["experiment__partition_method__name"]
    search_fields = ["experiment__folder_name"]
    readonly_fields = [
        "experiment", "diagonal_mean", "diagonal_std", "diagonal_min", "diagonal_max",
        "row_sum_min", "row_sum_max", "row_sum_mean", "row_sum_std",
        "col_sum_min", "col_sum_max", "col_sum_mean",
        "n_states_visited", "n_states_empty", "fraction_visited", 
        "eigenvalue_2", "spectral_gap", "matrix_bucket_path"
    ]
    
    fieldsets = (
        ("Expérience", {"fields": ["experiment"]}),
        ("Diagonale (P rester)", {
            "fields": ["diagonal_mean", "diagonal_std", "diagonal_min", "diagonal_max"]
        }),
        ("Sommes des lignes", {
            "fields": ["row_sum_mean", "row_sum_std", "row_sum_min", "row_sum_max"]
        }),
        ("Sommes des colonnes", {
            "fields": ["col_sum_mean", "col_sum_min", "col_sum_max"]
        }),
        ("États", {
            "fields": ["n_states_visited", "n_states_empty", "fraction_visited"]
        }),
        ("Spectre", {
            "fields": ["eigenvalue_2", "spectral_gap"]
        }),
        ("Stockage", {
            "fields": ["matrix_bucket_path"],
            "classes": ["collapse"]
        }),
    )
    
    def diag_mean_display(self, obj):
        val = obj.diagonal_mean
        if val > 0.7:
            color = "green"
        elif val > 0.5:
            color = "orange"
        else:
            color = "red"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, f'{val:.4f}'
        )
    diag_mean_display.short_description = "P(rester)"
    
    def spectral_gap_display(self, obj):
        val = obj.spectral_gap or 0
        if val > 0.3:
            color = "green"
        elif val > 0.1:
            color = "orange"
        else:
            color = "red"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, f'{val:.4f}'
        )
    spectral_gap_display.short_description = "Spectral gap"
    
    def has_add_permission(self, request):
        return False  # Les matrices s'ajoutent automatiquement


# ═══════════════════════════════════════════════════════════════
# RSD Results
# ═══════════════════════════════════════════════════════════════

class RSDResultAdmin(admin.ModelAdmin):
    list_display = ["experiment", "source_display", "rsd_initial_display", "rsd_final_display", "mixing_time_50"]
    list_filter = ["source", "experiment__partition_method__name", "created_at"]
    search_fields = ["experiment__folder_name"]
    readonly_fields = ["experiment", "rsd_curve", "entropy_curve", "created_at"]
    
    def source_display(self, obj):
        color = "green" if obj.source == "markov" else "blue"
        label = "Markov (prédit)" if obj.source == "markov" else "DEM (réel)"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{}</span>',
            color, label
        )
    source_display.short_description = "Source"
    
    def rsd_initial_display(self, obj):
        return f"{obj.rsd_initial:.2f}%"
    rsd_initial_display.short_description = "RSD initial"
    
    def rsd_final_display(self, obj):
        val = obj.rsd_final
        if val < 0.1:
            color = "green"
        elif val < 0.5:
            color = "orange"
        else:
            color = "red"
        return format_html(
            '<span style="color: {}; font-weight: bold;">{:.2f}%</span>',
            color, val
        )
    rsd_final_display.short_description = "RSD final"


# ═══════════════════════════════════════════════════════════════
# ENREGISTREMENT
# ═══════════════════════════════════════════════════════════════

admin.site.register(PartitionMethod, PartitionMethodAdmin)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(TransitionMatrix, TransitionMatrixAdmin)
admin.site.register(RSDResult, RSDResultAdmin)

#!/usr/bin/env python
"""
Script pour réinitialiser la base de données et recharger depuis le bucket.

Usage:
    python reset_db.py --full     # Réinitialiser complètement + recharger
    python reset_db.py --dry-run  # Juste lister sans importer
    python reset_db.py --empty    # Juste vider la DB
"""

import os
import sys
import django
import argparse
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dem_mcm.settings')
django.setup()

from django.core.management import call_command
from markov.models import (
    PartitionMethod, Experiment, TransitionMatrix, RSDResult
)


def empty_database():
    """Vide toutes les tables."""
    print("🗑️  Vidage complètement la base de données...")
    
    # Supprimer les enregistrements (pas les tables)
    print("   Suppression des RSDResult...")
    RSDResult.objects.all().delete()
    
    print("   Suppression des TransitionMatrix...")
    TransitionMatrix.objects.all().delete()
    
    print("   Suppression des Experiment...")
    Experiment.objects.all().delete()
    
    print("   Suppression des PartitionMethod...")
    PartitionMethod.objects.all().delete()
    
    print("✅ Base de données vidée\n")


def populate_partition_methods():
    """Crée les enregistrements de méthodes de partitionnement."""
    print("📝 Création des méthodes de partitionnement...")
    
    methods_data = [
        ("cartesian", "Cartésien", "Grille régulière 3D (nx, ny, nz)"),
        ("cylindrical", "Cylindrique", "Partitionnement cylindrique optimal pour mélangeurs"),
        ("voronoi", "Voronoï", "Cellules de Voronoï autour de centres K-means"),
        ("quantile", "Quantile", "Grille basée sur quantiles (distribution uniforme)"),
        ("octree", "Octree", "Partitionnement adaptatif récursif (plus dense où il y a plus)"),
        ("physics", "Physics-Aware", "Partitionnement basé sur la dynamique physique"),
        ("adaptive", "Adaptatif", "Deux zones (haute/basse) avec partitionnements différents"),
        ("multizone", "Multi-Zones", "N zones avec partitionnements différents (flexible)"),
        ("single", "Single Cell", "Une seule cellule pour tout le domaine (cas trivial)"),
    ]
    
    for name, label, description in methods_data:
        method, created = PartitionMethod.objects.get_or_create(
            name=name,
            defaults={
                'description': description,
                'label': label,
                'parameters': {}
            }
        )
        status = "✅ créée" if created else "⚠️  existante"
        print(f"   {name}: {status}")
    
    print(f"✅ {len(methods_data)} méthodes enregistrées\n")


def sync_bucket(dry_run=False):
    """Synchronise depuis le bucket HuggingFace."""
    print("🔄 Synchronisation depuis le bucket HuggingFace...")
    if dry_run:
        print("   (Mode dry-run - juste lister)\n")
        call_command('sync_bucket', dry_run=True, verbosity=2)
    else:
        print()
        call_command('sync_bucket', verbosity=2)


def show_summary():
    """Affiche un résumé de la base de données."""
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE LA BASE DE DONNÉES")
    print("="*60)
    
    methods_count = PartitionMethod.objects.count()
    experiments_count = Experiment.objects.count()
    matrices_count = TransitionMatrix.objects.count()
    
    print(f"✅ Méthodes de partitionnement: {methods_count}")
    print(f"✅ Expériences: {experiments_count}")
    print(f"✅ Matrices de transition: {matrices_count}")
    
    if experiments_count > 0:
        print("\n📂 Expériences par méthode:")
        for method in PartitionMethod.objects.all():
            exp_count = method.experiments.count()
            if exp_count > 0:
                print(f"   {method.name}: {exp_count} expériences")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Réinitialiser et recharger la base de données')
    parser.add_argument('--full', action='store_true', help='Vider + recharger (default)')
    parser.add_argument('--dry-run', action='store_true', help='Juste lister sans importer')
    parser.add_argument('--empty', action='store_true', help='Juste vider la DB')
    
    args = parser.parse_args()
    
    # Si aucune option, par défaut --full
    if not (args.full or args.dry_run or args.empty):
        args.full = True
    
    try:
        if args.empty:
            # Juste vider
            empty_database()
            populate_partition_methods()
            show_summary()
        
        elif args.dry_run:
            # Juste lister
            sync_bucket(dry_run=True)
        
        else:  # args.full
            # Vider complètement + recharger
            empty_database()
            populate_partition_methods()
            sync_bucket(dry_run=False)
            show_summary()
    
    except Exception as e:
        print(f"\n❌ Erreur: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

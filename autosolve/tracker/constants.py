# SPDX-FileCopyrightText: 2024-2025 AutoSolve Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Shared constants for AutoSolve tracking system.
Single source of truth for all configuration values.
"""

# ═══════════════════════════════════════════════════════════════════════════
# REGION DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

REGIONS = [
    'top-left', 'top-center', 'top-right',
    'mid-left', 'center', 'mid-right',
    'bottom-left', 'bottom-center', 'bottom-right'
]

EDGE_REGIONS = ['top-left', 'top-right', 'bottom-left', 'bottom-right']
CENTER_REGIONS = ['top-center', 'center', 'bottom-center']


# ═══════════════════════════════════════════════════════════════════════════
# TIERED SETTINGS (for retry/adaptation)
# ═══════════════════════════════════════════════════════════════════════════

TIERED_SETTINGS = {
    'ultra_aggressive': {
        'pattern_size': 31,
        'search_size': 151,
        'correlation': 0.45,
        'threshold': 0.15,
        'motion_model': 'Affine',
    },
    'aggressive': {
        'pattern_size': 27,
        'search_size': 130,
        'correlation': 0.50,
        'threshold': 0.12,
        'motion_model': 'Affine',
    },
    'moderate': {
        'pattern_size': 21,
        'search_size': 100,
        'correlation': 0.60,
        'threshold': 0.20,
        'motion_model': 'Affine',
    },
    'balanced': {
        'pattern_size': 15,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'selective': {
        'pattern_size': 13,
        'search_size': 61,
        'correlation': 0.75,
        'threshold': 0.40,
        'motion_model': 'LocRot',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PRETRAINED DEFAULTS (by resolution/fps)
# ═══════════════════════════════════════════════════════════════════════════

PRETRAINED_DEFAULTS = {
    'HD_24fps': {
        'pattern_size': 17,
        'search_size': 91,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    'HD_30fps': {
        'pattern_size': 15,
        'search_size': 71,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'HD_60fps': {
        'pattern_size': 13,
        'search_size': 51,
        'correlation': 0.72,
        'threshold': 0.35,
        'motion_model': 'LocRot',
    },
    '4K_24fps': {
        'pattern_size': 31,
        'search_size': 151,
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'Affine',
    },
    '4K_30fps': {
        'pattern_size': 29,
        'search_size': 141,
        'correlation': 0.65,
        'threshold': 0.25,
        'motion_model': 'LocRot',
    },
    '4K_60fps': {
        'pattern_size': 25,
        'search_size': 121,
        'correlation': 0.68,
        'threshold': 0.28,
        'motion_model': 'LocRot',
    },
    'SD_24fps': {
        'pattern_size': 13,
        'search_size': 81,
        'correlation': 0.70,
        'threshold': 0.30,
        'motion_model': 'LocRot',
    },
    'SD_30fps': {
        'pattern_size': 11,
        'search_size': 61,
        'correlation': 0.72,
        'threshold': 0.35,
        'motion_model': 'Loc',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FOOTAGE TYPE ADJUSTMENTS
# ═══════════════════════════════════════════════════════════════════════════

FOOTAGE_TYPE_ADJUSTMENTS = {
    'AUTO': {},
    'INDOOR': {
        'correlation': 0.72,
        'threshold': 0.30,
    },
    'OUTDOOR': {
        'dead_zones': ['top-center'],
        'threshold': 0.25,
    },
    'DRONE': {
        'search_size_mult': 1.3,
        'pattern_size_mult': 1.2,
        'correlation': 0.60,
        'threshold': 0.20,
        'dead_zones': ['top-left', 'top-center', 'top-right'],
        'motion_model': 'Affine',
    },
    'HANDHELD': {
        'search_size_mult': 1.2,
        'correlation': 0.65,
        'motion_model': 'LocRot',
    },
    'GIMBAL': {
        'search_size_mult': 0.9,
        'correlation': 0.72,
        'threshold': 0.32,
    },
    'ACTION': {
        'search_size_mult': 1.5,
        'pattern_size_mult': 1.3,
        'correlation': 0.50,
        'threshold': 0.15,
        'motion_model': 'Affine',
    },
    'VFX': {
        'correlation': 0.75,
        'threshold': 0.35,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT SETTINGS (cold start fallback)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SETTINGS = {
    'pattern_size': 15,
    'search_size': 71,
    'correlation': 0.70,
    'threshold': 0.30,
    'motion_model': 'LocRot',
}

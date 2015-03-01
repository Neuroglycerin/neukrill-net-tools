#!/usr/bin/env python

taxonomy = {
    "no_class": {
        "artifacts": {},
        "artifacts_edge": {}
    },
    "plankton": {
        "unknown": {
            "unknown_unclassified": {},
            "unknown_sticks": {},
            "unknown_blobs_and_smudges": {}
        },
        "trichodesmium": {
            "trichodesmium_tuft": {},
            "trichodesmium_bowtie": {},
            "trichodesmium_puff": {},
            "trichodesmium_multiple": {}
        },
        "protists": {
            "acantharia": {
                "acantharia_protist": {},
                "acantharia_protist_big_center": {},
                "acantharia_protist_halo": {}
            },
            "protist_noctiluca": {},
            "sub_protists": {
                "protist_other": {},
                "protist_star": {},
                "protist_fuzzy_olive": {},
                "protist_dark_center": {}
            },
            "radiolarian": {
                "radiolarian_colony": {},
                "radiolarian_chain": {}
            }
        },
        "fish": {
            "fish_larvae_very_thin_body": {},
            "fish_larvae_thin_body": {},
            "fish_larvae_medium_body": {},
            "fish_larvae_deep_body": {},
            "fish_larvae_myctophids": {},
            "fish_larvae_leptocephali": {}
        },
        "chaetognaths": {
            "chaetognath_other": {},
            "chaetognath_sagitta": {},
            "chaetognath_non_sagitta": {}
        },
        "other_invert_larvae": {
            "invertebrate_larvae_other_A": {},
            "invertebrate_larvae_other_B": {},
            "tornaria_acorn_worm_larvae": {},
            "trochophore_larvae": {},
            "echinoderm": {
                "pluteus": {
                    "echinoderm_larva_pluteus_early": {},
                    "echinoderm_larva_pluteus_urchin": {},
                    "echinoderm_larva_pluteus_typeC": {},
                    "echinoderm_larva_pluteus_brittlestar": {},
                    "echinopluteus": {}
                },
                "seastar": {
                    "echinoderm_larva_seastar_bipinnaria": {},
                    "echinoderm_larva_seastar_brachiolaria": {}
                },
                "echinoderm_seacucumber_auricularia_larva": {}
            }
        },
        "diatoms": {
            "diatom_chain_string": {},
            "diatom_chain_tube": {}
        },
        "polychaete": {},
        "detritus": {
            "fecal_pellet": {},
            "detritus_blob": {},
            "detritus_other": {},
            "detritus_filamentous": {}
        },
        "gastropods": {
            "heteropod": {},
            "pteropods": {
                "pteropod_butterfly": {},
                "pteropod_triangle": {},
                "pteropod_theco_dev_seq": {}
            }
        },
        "crustaceans": {
            "crustacean_other": {},
            "stomatopod": {},
            "amphipods": {},
            "shrimp_like": {
                "shrimp-like_other": {},
                "euphausiids": {
                    "euphausiids": {},
                    "euphausiids_young": {}
                },
                "decapods": {
                    "decapods": {},
                    "shrimp_zoea": {},
                    "shrimp_caridean": {},
                    "shrimp_sergestidae": {}
                }
            },
            "copepods": {
                "cyclopoid_copepods": {
                    "copepod_cyclopoid_copilia": {},
                    "oithona": {
                        "copepod_cyclopoid_oithona": {},
                        "copepod_cyclopoid_oithona_eggs": {}
                    }
                },
                "calanoid": {
                    "copepod_calanoid": {},
                    "copepod_other": {},
                    "copepod_calanoid_small_longantennae": {},
                    "copepod_calanoid_frillyAntennae": {},
                    "copepod_calanoid_flatheads": {},
                    "copepod_calanoid_eggs": {},
                    "copepod_calanoid_octomoms": {},
                    "copepod_calanoid_large": {},
                    "copepod_calanoid_large_side_antennatucked": {},
                    "copepod_calanoid_eucalanus": {}
                }
            }
        },
        "chordate_type1": {},
        "gelatinous zooplankton": {
            "jellies_tentacles": {},
            "ephyra": {},
            "ctenophores": {
                "ctenophore_cestid": {},
                "cydippid": {
                    "ctenophore_cydippid_tentacles": {},
                    "ctenophore_cydippid_no_tentacles": {}
                },
                "ctenophore_lobate": {}
            },
            "pelagic_tunicates": {
                "tunicate": {
                    "tunicate_doliolid": {},
                    "tunicate_doliolid_nurse": {},
                    "tunicate_salp": {},
                    "tunicate_salp_chains": {},
                    "tunicate_partial": {}
                },
                "appendicularians": {
                    "appendicularian_fritillaridae": {},
                    "appendicularian_s_shape": {},
                    "appendicularian_slight_curve": {},
                    "appendicularian_straight": {}
                }
            },
            "siphonophores": {
                "siphonophore_other_parts": {},
                "siphonophore_partial": {},
                "physonect": {
                    "siphonophore_physonect": {},
                    "siphonophore_physonect_young": {}
                },
                "calycophoran_siphonophores": {
                    "siphonophore_calycophoran_abylidae": {},
                    "rocketship": {
                        "siphonophore_calycophoran_rocketship_adult": {},
                        "siphonophore_calycophoran_rocketship_young": {}
                    },
                    "sphaeronectes": {
                        "siphonophore_calycophoran_sphaeronectes": {},
                        "siphonophore_calycophoran_sphaeronectes_young": {},
                        "siphonophore_calycophoran_sphaeronectes_stem": {}
                    }
                }
            },
            "hydromedusae": {
                "sub_hydromedusae1": {
                    "hydromedusae_liriope": {},
                    "hydromedusae_aglaura": {},
                    "hydromedusae_haliscera": {},
                    "hydromedusae_haliscera_small_sideview": {}
                },
                "sub_hydromedusae2": {
                    "hydromedusae_narcomedusae": {},
                    "hydromedusae_narco_dark": {},
                    "hydromedusae_solmaris": {},
                    "hydromedusae_solmundella": {},
                    "hydromedusae_narco_young": {}
                },
                "other_hydromedusae": {
                    "hydromedusae_partial_dark": {},
                    "hydromedusae_typeD_bell_and_tentacles": {},
                    "hydromedusae_typeD": {},
                    "hydromedusae_bell_and_tentacles": {},
                    "hydromedusae_typeE": {},
                    "hydromedusae_typeF": {},
                    "hydromedusae_shapeA": {},
                    "hydromedusae_shapeA_sideview_small": {},
                    "hydromedusae_sideview_big": {},
                    "hydromedusae_shapeB": {},
                    "hydromedusae_h15": {},
                    "hydromedusae_other": {}
                }
            }
        }
    }
}

superclasses = {}


def generate_superclasses(taxonomy, current_parents):
    if not taxonomy:
        superclasses[current_parents[-1]] = current_parents[0:-1]
    for parent, children in taxonomy.iteritems():
        current_parents.append(parent)
        generate_superclasses(children, current_parents)
        current_parents.pop()

generate_superclasses(taxonomy, [])


class TaxonomyLayer(object):
    """A hierarchical representation of plankton species"""
    def __init__(self, depth):
        self.depth = depth

    def __getitem__(self, item):
        if self.depth > len(superclasses[item]):
            return item
        return superclasses[item][self.depth - 1]

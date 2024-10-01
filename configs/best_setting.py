BEST_AGENT = {

    "SimEnv3-random-SGaussian": {
        "SQL": {
            " --param ": [
                7
            ],
            " --pi_lr ": [
                0.003
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                5
            ],
            " --pi_lr ": [
                0.0001
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "IQL": {
            " --param ": [
                23
            ],
            " --pi_lr ": [
                3e-05
            ]
        },
        "TAWAC": {
            " --param ": [
                19
            ],
            " --pi_lr ": [
                3e-05
            ]
        },
    },
    "Ant-expert-Student": {
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.01
            ]
        },
    },

    "HTqG-KL": {
        "SimEnv3-random-qGaussian": {
            "FTT": {
                " --param ": [
                    3
                ],
                " --pi_lr ": [
                    0.0003
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "HalfCheetah-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "HalfCheetah-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "HalfCheetah-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    1
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Hopper-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Hopper-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Hopper-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Walker2d-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    1
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Walker2d-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
        "Walker2d-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ]
            }
        },
    },
    "SG-KL": {
        "SimEnv3-random-qGaussian": {
            "FTT": {
                " --param ": [
                    9
                ],
                " --pi_lr ": [
                    3e-05
                ],
                " --tau ": [
                    0.5
                ]
            }
        },
        "HalfCheetah-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "HalfCheetah-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "HalfCheetah-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    1
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.5
                ]
            }
        },
        "Hopper-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "Hopper-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "Hopper-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    4
                ],
                " --pi_lr ": [
                    0.0003
                ],
                " --tau ": [
                    0.5
                ]
            }
        },
        "Walker2d-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "Walker2d-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "Walker2d-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
    },
    "HTqG-SPOT": {
        "SimEnv3-random-qGaussian": {
            "FTT": {
                " --param ": [
                    9
                ],
                " --pi_lr ": [
                    3e-05
                ],
                " --tau ": [
                    0.5
                ]
            }
        },
        "HalfCheetah-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "HalfCheetah-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "HalfCheetah-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "Hopper-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    0
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "Hopper-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    2
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "Hopper-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    5
                ],
                " --pi_lr ": [
                    0.0003
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
        "Walker2d-medexp-qGaussian": {
            "FTT": {
                " --param ": [
                    3
                ],
                " --pi_lr ": [
                    0.0003
                ],
                " --tau ": [
                    1.0
                ]
            }
        },
        "Walker2d-medium-qGaussian": {
            "FTT": {
                " --param ": [
                    1
                ],
                " --pi_lr ": [
                    0.001
                ],
                " --tau ": [
                    0.5
                ]
            }
        },
        "Walker2d-medrep-qGaussian": {
            "FTT": {
                " --param ": [
                    5
                ],
                " --pi_lr ": [
                    0.0003
                ],
                " --tau ": [
                    0.01
                ]
            }
        },
    },

    "HalfCheetah-medexp-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medexp-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medexp-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medexp-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "HalfCheetah-medexp-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medium-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medium-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medium-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medium-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "HalfCheetah-medium-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                5
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "HalfCheetah-medrep-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        }
    },
    "HalfCheetah-medrep-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "HalfCheetah-medrep-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "HalfCheetah-medrep-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        }
    },
    "HalfCheetah-medrep-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medexp-SGaussian": {
        "SQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medexp-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Hopper-medexp-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medexp-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medexp-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medium-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medium-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medium-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medium-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                11
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Hopper-medium-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medrep-SGaussian": {
        "SQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medrep-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medrep-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Hopper-medrep-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Hopper-medrep-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Walker2d-medexp-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Walker2d-medexp-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Walker2d-medexp-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Walker2d-medexp-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Walker2d-medexp-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Walker2d-medium-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Walker2d-medium-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Walker2d-medium-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Walker2d-medium-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.01
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Walker2d-medium-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "InAC": {
            " --tau ": [
                0.33
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                1.0
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                2
            ],
            " --pi_lr ": [
                0.0001
            ]
        }
    },
    "Walker2d-medrep-SGaussian": {
        "SQL": {
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                5.0
            ],
            " --expectile ": [
                5.0
            ]
        },
        "XQL": {
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.0002
            ],
            " --tau ": [
                2.0
            ],
            " --expectile ": [
                2.0
            ]
        },
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        }
    },
    "Walker2d-medrep-Beta": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Walker2d-medrep-Student": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        }
    },
    "Walker2d-medrep-HTqGaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        }
    },
    "Walker2d-medrep-Gaussian": {
        "IQL": {
            " --expectile ": [
                0.7
            ],
            " --tau ": [
                0.3333333333333333
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "InAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "TAWAC": {
            " --tau ": [
                0.5
            ],
            " --param ": [
                0
            ],
            " --pi_lr ": [
                0.001
            ]
        },
        "AWAC": {
            " --tau ": [
                0.1
            ],
            " --param ": [
                1
            ],
            " --pi_lr ": [
                0.0003
            ]
        },
        "TD3BC": {
            " --tau ": [
                2.5
            ],
            " --param ": [
                3
            ],
            " --pi_lr ": [
                0.003
            ]
        }
    }
}
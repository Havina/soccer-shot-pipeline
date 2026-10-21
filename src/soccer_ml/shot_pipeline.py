
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

GOAL_CENTER_X = 0.0
GOAL_CENTER_Y = 34.0


BASELINE_FEATURES = [
    "distance_to_goal",
    "shot_angle",
]


ENHANCED_FEATURES = [
    "distance_to_goal",
    "shot_angle",
    "nearest_defender_distance",
    "defenders_within_5m",
]


TEAM_NAME_TO_CODE = {
    "Arsenal": "ARSENAL",
    "Crystal Palace": "CRYSTAL_PALACE",
}


OPPONENT_CODE = {
    "ARSENAL": "CRYSTAL_PALACE",
    "CRYSTAL_PALACE": "ARSENAL",
}


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class ShotAnalysisResult:

    shot_id: str

    has_homography: bool
    has_automatic_context: bool

    shooting_team: Optional[str] = None
    defending_team: Optional[str] = None

    shot_pitch_x: Optional[float] = None
    shot_pitch_y: Optional[float] = None

    distance_to_goal: Optional[float] = None
    shot_angle_radians: Optional[float] = None
    shot_angle_degrees: Optional[float] = None

    detected_outfield_players: Optional[int] = None
    detected_defenders: Optional[int] = None
    detected_attackers: Optional[int] = None

    shooter_detection_id: Optional[int] = None
    shooter_distance_to_shot: Optional[float] = None
    shooter_match_quality: Optional[str] = None

    nearest_defender_distance: Optional[float] = None
    defenders_within_3m: Optional[int] = None
    defenders_within_5m: Optional[int] = None
    defenders_within_10m: Optional[int] = None
    pressure_score: Optional[float] = None

    nearest_teammate_distance: Optional[float] = None
    teammates_within_5m: Optional[int] = None
    teammates_within_10m: Optional[int] = None

    goalkeeper_detection_id: Optional[int] = None
    goalkeeper_pitch_x: Optional[float] = None
    goalkeeper_pitch_y: Optional[float] = None
    goalkeeper_depth_m: Optional[float] = None
    goalkeeper_lateral_offset_m: Optional[float] = None
    goalkeeper_shot_line_offset_m: Optional[float] = None

    defenders_between_shot_and_goal: Optional[int] = None
    shot_lane_defenders_1m: Optional[int] = None
    shot_lane_defenders_2m: Optional[int] = None
    shot_lane_defenders_3m: Optional[int] = None

    baseline_xg: Optional[float] = None
    automatic_context_xg: Optional[float] = None

    pipeline_status: str = "UNPROCESSED"


# ============================================================
# SAFE TYPE HELPERS
# ============================================================

def _safe_float(value):

    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


def _safe_int(value):

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# MAIN PIPELINE CLASS
# ============================================================

class SoccerShotPipeline:
    """
    Reusable Soccer ML shot-analysis pipeline.

    Current pipeline stages:
        1. shot metadata lookup
        2. homography lookup
        3. player detection retrieval
        4. image-to-pitch projection
        5. player filtering
        6. automatic shot-context extraction
        7. goalkeeper context
        8. shot-lane context
        9. baseline xG
       10. automatic-context enhanced xG
    """

    def __init__(
        self,
        project_root
    ):

        self.project_root = Path(
            project_root
        ).resolve()

        self.processed_dir = (
            self.project_root
            / "data"
            / "processed"
        )

        self.models_dir = (
            self.project_root
            / "models"
        )


        # ----------------------------------------------------
        # Artifact paths
        # ----------------------------------------------------
        self.homography_path = (
            self.processed_dir
            / "notebook27_semi_auto_homographies.npz"
        )

        self.detections_path = (
            self.processed_dir
            / "notebook28_final_player_detections.csv"
        )

        self.shot_dataset_path = (
            self.processed_dir
            / "crystal_palace_arsenal_video_shot_dataset.csv"
        )

        self.baseline_model_path = (
            self.models_dir
            / "notebook24_baseline_xg_model.joblib"
        )

        self.enhanced_model_path = (
            self.models_dir
            / "notebook24_tracking_enhanced_xg_model.joblib"
        )


        # ----------------------------------------------------
        # Validate required artifacts
        # ----------------------------------------------------
        required = [
            self.homography_path,
            self.detections_path,
            self.shot_dataset_path,
            self.baseline_model_path,
            self.enhanced_model_path,
        ]

        missing = [
            path
            for path in required
            if not path.exists()
        ]

        if missing:

            missing_text = "\n".join(
                str(path)
                for path in missing
            )

            raise FileNotFoundError(
                "Required Soccer ML artifacts are missing:\n"
                + missing_text
            )


        # ----------------------------------------------------
        # Load reusable artifacts
        # ----------------------------------------------------
        self.homography_data = np.load(
            self.homography_path,
            allow_pickle=True
        )

        self.raw_detections = pd.read_csv(
            self.detections_path
        )

        self.shot_dataset = pd.read_csv(
            self.shot_dataset_path
        )

        self.baseline_model = joblib.load(
            self.baseline_model_path
        )

        self.enhanced_model = joblib.load(
            self.enhanced_model_path
        )


        # ----------------------------------------------------
        # Validate trained-model contracts
        # ----------------------------------------------------
        baseline_features = list(
            self.baseline_model.feature_names_in_
        )

        enhanced_features = list(
            self.enhanced_model.feature_names_in_
        )

        if baseline_features != BASELINE_FEATURES:

            raise ValueError(
                "Unexpected baseline-model feature schema: "
                f"{baseline_features}"
            )

        if enhanced_features != ENHANCED_FEATURES:

            raise ValueError(
                "Unexpected enhanced-model feature schema: "
                f"{enhanced_features}"
            )


    # ========================================================
    # PUBLIC METADATA
    # ========================================================

    @property
    def available_shots(self):

        return sorted(
            self.shot_dataset[
                "shot_id"
            ].unique()
        )


    # ========================================================
    # TEAM NORMALIZATION
    # ========================================================

    @staticmethod
    def normalize_team_name(team_name):

        if pd.isna(team_name):
            return None

        return TEAM_NAME_TO_CODE.get(
            team_name,
            str(team_name)
            .upper()
            .replace(" ", "_")
        )


    # ========================================================
    # SHOT INPUT RETRIEVAL
    # ========================================================

    def get_shot_inputs(
        self,
        shot_id
    ):

        shot_rows = self.shot_dataset[
            self.shot_dataset[
                "shot_id"
            ] == shot_id
        ].copy()


        if len(shot_rows) == 0:

            raise ValueError(
                f"Unknown shot_id: {shot_id}"
            )


        if len(shot_rows) > 1:

            raise ValueError(
                "Duplicate shot metadata rows found for: "
                f"{shot_id}"
            )


        shot = shot_rows.iloc[0]


        shooting_team = (
            self.normalize_team_name(
                shot["shooting_team"]
            )
        )


        defending_team = (
            OPPONENT_CODE.get(
                shooting_team
            )
        )


        shot_pitch_x = float(
            shot["pitch_x"]
        )

        shot_pitch_y = float(
            shot["pitch_y"]
        )

        distance_to_goal = float(
            shot["distance_to_goal"]
        )

        shot_angle_radians = float(
            shot["shot_angle"]
        )

        shot_angle_degrees = float(
            np.degrees(
                shot_angle_radians
            )
        )


        has_homography = (
            shot_id
            in self.homography_data.files
        )


        if has_homography:

            homography = (
                self.homography_data[
                    shot_id
                ].copy()
            )

        else:

            homography = None


        detections = (
            self.raw_detections[
                self.raw_detections[
                    "shot_id"
                ] == shot_id
            ]
            .copy()
            .reset_index(drop=True)
        )


        if len(detections) == 0:

            raise ValueError(
                "No raw detections found for: "
                f"{shot_id}"
            )


        return {
            "shot_id":
                shot_id,

            "game_time":
                shot["game_time"],

            "shot_frame":
                int(
                    shot["shot_frame"]
                ),

            "analysis_frame":
                int(
                    shot["analysis_frame"]
                ),

            "shooting_team":
                shooting_team,

            "defending_team":
                defending_team,

            "shot_pitch_x":
                shot_pitch_x,

            "shot_pitch_y":
                shot_pitch_y,

            "distance_to_goal":
                distance_to_goal,

            "shot_angle_radians":
                shot_angle_radians,

            "shot_angle_degrees":
                shot_angle_degrees,

            "goal":
                int(
                    shot["goal"]
                ),

            "has_homography":
                has_homography,

            "homography":
                homography,

            "detections":
                detections,
        }


    # ========================================================
    # IMAGE -> PITCH PROJECTION
    # ========================================================

    @staticmethod
    def project_image_point(
        homography,
        x,
        y
    ):

        point = np.array(
            [
                float(x),
                float(y),
                1.0,
            ],
            dtype=np.float64
        )


        mapped = (
            homography
            @ point
        )

        w = mapped[2]


        if (
            not np.isfinite(w)
            or
            abs(w) < 1e-10
        ):

            return (
                np.nan,
                np.nan
            )


        pitch_x = (
            mapped[0]
            / w
        )

        pitch_y = (
            mapped[1]
            / w
        )


        return (
            float(pitch_x),
            float(pitch_y)
        )


    def project_and_filter_detections(
        self,
        shot_bundle
    ):

        detections = (
            shot_bundle[
                "detections"
            ].copy()
        )


        detections["pitch_x"] = np.nan
        detections["pitch_y"] = np.nan


        # ----------------------------------------------------
        # No homography
        # ----------------------------------------------------
        if not shot_bundle[
            "has_homography"
        ]:

            detections[
                "inside_pitch_bounds"
            ] = pd.Series(
                pd.NA,
                index=detections.index,
                dtype="boolean"
            )

            detections[
                "has_projection"
            ] = False

            detections[
                "valid_player_position"
            ] = False

            detections[
                "usable_outfield"
            ] = False

            detections[
                "usable_goalkeeper"
            ] = False

            detections[
                "edge_player_unresolved"
            ] = False

            detections[
                "analysis_status"
            ] = "NO_HOMOGRAPHY"

            return detections


        # ----------------------------------------------------
        # Project detections
        # ----------------------------------------------------
        H = shot_bundle[
            "homography"
        ]


        for idx, row in detections.iterrows():

            px, py = self.project_image_point(
                H,
                row["ground_x"],
                row["ground_y"]
            )

            detections.at[
                idx,
                "pitch_x"
            ] = px

            detections.at[
                idx,
                "pitch_y"
            ] = py


        detections[
            "has_projection"
        ] = (
            detections[
                "pitch_x"
            ].notna()
            &
            detections[
                "pitch_y"
            ].notna()
        )


        inside_mask = (
            detections[
                "pitch_x"
            ].between(
                0.0,
                PITCH_LENGTH,
                inclusive="both"
            )
            &
            detections[
                "pitch_y"
            ].between(
                0.0,
                PITCH_WIDTH,
                inclusive="both"
            )
        )


        detections[
            "inside_pitch_bounds"
        ] = (
            inside_mask.astype(
                "boolean"
            )
        )


        detections.loc[
            ~detections[
                "has_projection"
            ],
            "inside_pitch_bounds"
        ] = pd.NA


        detections[
            "valid_player_position"
        ] = (
            detections[
                "has_projection"
            ]
            &
            (
                detections[
                    "inside_pitch_bounds"
                ]
                == True
            )
            &
            detections[
                "role"
            ].isin(
                [
                    "OUTFIELD_PLAYER",
                    "GOALKEEPER",
                    "EDGE_TRUNCATED",
                ]
            )
        )


        detections[
            "usable_outfield"
        ] = (
            detections[
                "valid_player_position"
            ]
            &
            (
                detections[
                    "role"
                ]
                == "OUTFIELD_PLAYER"
            )
            &
            detections[
                "team"
            ].isin(
                [
                    "ARSENAL",
                    "CRYSTAL_PALACE",
                ]
            )
        )


        detections[
            "usable_goalkeeper"
        ] = (
            detections[
                "valid_player_position"
            ]
            &
            (
                detections[
                    "role"
                ]
                == "GOALKEEPER"
            )
        )


        detections[
            "edge_player_unresolved"
        ] = (
            detections[
                "valid_player_position"
            ]
            &
            (
                detections[
                    "role"
                ]
                == "EDGE_TRUNCATED"
            )
        )


        detections[
            "analysis_status"
        ] = "EXCLUDED"


        detections.loc[
            detections[
                "has_projection"
            ]
            &
            (
                detections[
                    "inside_pitch_bounds"
                ]
                == False
            ),
            "analysis_status"
        ] = "OUTSIDE_PITCH"


        detections.loc[
            detections[
                "usable_outfield"
            ],
            "analysis_status"
        ] = "USABLE_OUTFIELD"


        detections.loc[
            detections[
                "usable_goalkeeper"
            ],
            "analysis_status"
        ] = "USABLE_GOALKEEPER"


        detections.loc[
            detections[
                "edge_player_unresolved"
            ],
            "analysis_status"
        ] = "EDGE_TEAM_UNRESOLVED"


        return detections


    # ========================================================
    # CONTEXT HELPERS
    # ========================================================

    @staticmethod
    def get_shooter_match_quality(
        distance
    ):

        if (
            distance is None
            or
            not np.isfinite(distance)
        ):

            return None


        if distance <= 1.5:
            return "STRONG"

        if distance <= 3.0:
            return "PLAUSIBLE"

        if distance <= 5.0:
            return "WEAK"

        return "POOR"


    @staticmethod
    def point_to_line_distance(
        point_x,
        point_y,
        line_x1,
        line_y1,
        line_x2,
        line_y2
    ):

        dx = (
            line_x2
            -
            line_x1
        )

        dy = (
            line_y2
            -
            line_y1
        )


        line_length = np.sqrt(
            dx ** 2
            +
            dy ** 2
        )


        if line_length <= 0:
            return np.nan


        numerator = abs(
            dy * point_x
            -
            dx * point_y
            +
            line_x2 * line_y1
            -
            line_y2 * line_x1
        )


        return float(
            numerator
            /
            line_length
        )


    # ========================================================
    # AUTOMATIC CONTEXT
    # ========================================================

    def compute_automatic_context(
        self,
        shot_bundle,
        projected_detections
    ):

        shot_id = (
            shot_bundle[
                "shot_id"
            ]
        )


        # ----------------------------------------------------
        # No homography
        # ----------------------------------------------------
        if not shot_bundle[
            "has_homography"
        ]:

            return {
                "shot_id":
                    shot_id,

                "auto_context_available":
                    False,

                "detected_outfield_players":
                    0,

                "detected_defenders":
                    0,

                "detected_attackers":
                    0,

                "shooter_detection_id":
                    None,

                "shooter_distance_to_shot":
                    None,

                "shooter_match_quality":
                    None,

                "nearest_defender_distance":
                    None,

                "defenders_within_3m":
                    None,

                "defenders_within_5m":
                    None,

                "defenders_within_10m":
                    None,

                "pressure_score":
                    None,

                "nearest_teammate_distance":
                    None,

                "teammates_within_5m":
                    None,

                "teammates_within_10m":
                    None,

                "goalkeeper_detection_id":
                    None,

                "goalkeeper_pitch_x":
                    None,

                "goalkeeper_pitch_y":
                    None,

                "goalkeeper_depth_m":
                    None,

                "goalkeeper_lateral_offset_m":
                    None,

                "goalkeeper_shot_line_offset_m":
                    None,

                "defenders_between_shot_and_goal":
                    None,

                "shot_lane_defenders_1m":
                    None,

                "shot_lane_defenders_2m":
                    None,

                "shot_lane_defenders_3m":
                    None,

                "nearest_shot_lane_defender_offset_m":
                    None,
            }


        shot_x = float(
            shot_bundle[
                "shot_pitch_x"
            ]
        )

        shot_y = float(
            shot_bundle[
                "shot_pitch_y"
            ]
        )

        shooting_team = (
            shot_bundle[
                "shooting_team"
            ]
        )

        defending_team = (
            shot_bundle[
                "defending_team"
            ]
        )


        outfield = projected_detections[
            projected_detections[
                "usable_outfield"
            ] == True
        ].copy()


        attackers = outfield[
            outfield[
                "team"
            ] == shooting_team
        ].copy()


        defenders = outfield[
            outfield[
                "team"
            ] == defending_team
        ].copy()


        detected_outfield_players = len(
            outfield
        )

        detected_attackers = len(
            attackers
        )

        detected_defenders = len(
            defenders
        )


        # ----------------------------------------------------
        # Defender distances
        # ----------------------------------------------------
        defender_distances = []


        for _, defender in defenders.iterrows():

            distance = np.sqrt(
                (
                    defender[
                        "pitch_x"
                    ]
                    -
                    shot_x
                ) ** 2
                +
                (
                    defender[
                        "pitch_y"
                    ]
                    -
                    shot_y
                ) ** 2
            )


            defender_distances.append(
                {
                    "detection_id":
                        int(
                            defender[
                                "detection_id"
                            ]
                        ),

                    "pitch_x":
                        float(
                            defender[
                                "pitch_x"
                            ]
                        ),

                    "pitch_y":
                        float(
                            defender[
                                "pitch_y"
                            ]
                        ),

                    "distance_to_shot":
                        float(
                            distance
                        ),
                }
            )


        defender_distance_df = pd.DataFrame(
            defender_distances
        )


        if len(
            defender_distance_df
        ) > 0:

            defender_distance_df = (
                defender_distance_df
                .sort_values(
                    "distance_to_shot"
                )
                .reset_index(
                    drop=True
                )
            )


            nearest_defender_distance = float(
                defender_distance_df[
                    "distance_to_shot"
                ].min()
            )


            defenders_within_3m = int(
                (
                    defender_distance_df[
                        "distance_to_shot"
                    ] <= 3.0
                ).sum()
            )


            defenders_within_5m = int(
                (
                    defender_distance_df[
                        "distance_to_shot"
                    ] <= 5.0
                ).sum()
            )


            defenders_within_10m = int(
                (
                    defender_distance_df[
                        "distance_to_shot"
                    ] <= 10.0
                ).sum()
            )


            pressure_score = int(
                defenders_within_10m
                +
                2
                *
                defenders_within_5m
            )

        else:

            nearest_defender_distance = None
            defenders_within_3m = None
            defenders_within_5m = None
            defenders_within_10m = None
            pressure_score = None


        # ----------------------------------------------------
        # Attacker distances / shooter
        # ----------------------------------------------------
        attacker_distances = []


        for _, attacker in attackers.iterrows():

            distance = np.sqrt(
                (
                    attacker[
                        "pitch_x"
                    ]
                    -
                    shot_x
                ) ** 2
                +
                (
                    attacker[
                        "pitch_y"
                    ]
                    -
                    shot_y
                ) ** 2
            )


            attacker_distances.append(
                {
                    "detection_id":
                        int(
                            attacker[
                                "detection_id"
                            ]
                        ),

                    "distance_to_shot":
                        float(
                            distance
                        ),
                }
            )


        attacker_distance_df = pd.DataFrame(
            attacker_distances
        )


        if len(
            attacker_distance_df
        ) > 0:

            attacker_distance_df = (
                attacker_distance_df
                .sort_values(
                    "distance_to_shot"
                )
                .reset_index(
                    drop=True
                )
            )


            shooter = (
                attacker_distance_df
                .iloc[0]
            )


            shooter_detection_id = int(
                shooter[
                    "detection_id"
                ]
            )


            shooter_distance_to_shot = float(
                shooter[
                    "distance_to_shot"
                ]
            )


            shooter_match_quality = (
                self.get_shooter_match_quality(
                    shooter_distance_to_shot
                )
            )


            teammates = (
                attacker_distance_df
                .iloc[1:]
                .copy()
            )


            if len(teammates) > 0:

                nearest_teammate_distance = float(
                    teammates[
                        "distance_to_shot"
                    ].min()
                )


                teammates_within_5m = int(
                    (
                        teammates[
                            "distance_to_shot"
                        ] <= 5.0
                    ).sum()
                )


                teammates_within_10m = int(
                    (
                        teammates[
                            "distance_to_shot"
                        ] <= 10.0
                    ).sum()
                )

            else:

                nearest_teammate_distance = None
                teammates_within_5m = 0
                teammates_within_10m = 0

        else:

            shooter_detection_id = None
            shooter_distance_to_shot = None
            shooter_match_quality = None

            nearest_teammate_distance = None
            teammates_within_5m = None
            teammates_within_10m = None


        # ----------------------------------------------------
        # Goalkeeper
        # ----------------------------------------------------
        goalkeepers = projected_detections[
            projected_detections[
                "usable_goalkeeper"
            ] == True
        ].copy()


        if len(goalkeepers) == 1:

            goalkeeper = (
                goalkeepers.iloc[0]
            )


            goalkeeper_detection_id = int(
                goalkeeper[
                    "detection_id"
                ]
            )


            goalkeeper_pitch_x = float(
                goalkeeper[
                    "pitch_x"
                ]
            )


            goalkeeper_pitch_y = float(
                goalkeeper[
                    "pitch_y"
                ]
            )


            goalkeeper_depth_m = (
                goalkeeper_pitch_x
            )


            goalkeeper_lateral_offset_m = abs(
                goalkeeper_pitch_y
                -
                GOAL_CENTER_Y
            )


            goalkeeper_shot_line_offset_m = (
                self.point_to_line_distance(
                    goalkeeper_pitch_x,
                    goalkeeper_pitch_y,
                    shot_x,
                    shot_y,
                    GOAL_CENTER_X,
                    GOAL_CENTER_Y
                )
            )

        else:

            goalkeeper_detection_id = None
            goalkeeper_pitch_x = None
            goalkeeper_pitch_y = None
            goalkeeper_depth_m = None
            goalkeeper_lateral_offset_m = None
            goalkeeper_shot_line_offset_m = None


        # ----------------------------------------------------
        # Shot-lane geometry
        # ----------------------------------------------------
        defenders_between_shot_and_goal = 0
        shot_lane_defenders_1m = 0
        shot_lane_defenders_2m = 0
        shot_lane_defenders_3m = 0

        lane_offsets = []


        vx = (
            GOAL_CENTER_X
            -
            shot_x
        )

        vy = (
            GOAL_CENTER_Y
            -
            shot_y
        )


        segment_length_sq = (
            vx ** 2
            +
            vy ** 2
        )


        if (
            len(
                defender_distance_df
            ) > 0
            and
            segment_length_sq > 0
        ):

            for _, defender in (
                defender_distance_df
                .iterrows()
            ):

                wx = (
                    defender[
                        "pitch_x"
                    ]
                    -
                    shot_x
                )

                wy = (
                    defender[
                        "pitch_y"
                    ]
                    -
                    shot_y
                )


                t = (
                    wx * vx
                    +
                    wy * vy
                ) / segment_length_sq


                projected_x = (
                    shot_x
                    +
                    t * vx
                )

                projected_y = (
                    shot_y
                    +
                    t * vy
                )


                lane_offset = np.sqrt(
                    (
                        defender[
                            "pitch_x"
                        ]
                        -
                        projected_x
                    ) ** 2
                    +
                    (
                        defender[
                            "pitch_y"
                        ]
                        -
                        projected_y
                    ) ** 2
                )


                between = (
                    0.0
                    <= t
                    <= 1.0
                )


                if between:

                    defenders_between_shot_and_goal += 1

                    lane_offsets.append(
                        float(
                            lane_offset
                        )
                    )


                    if lane_offset <= 1.0:
                        shot_lane_defenders_1m += 1

                    if lane_offset <= 2.0:
                        shot_lane_defenders_2m += 1

                    if lane_offset <= 3.0:
                        shot_lane_defenders_3m += 1


        if len(lane_offsets) > 0:

            nearest_shot_lane_defender_offset_m = float(
                min(
                    lane_offsets
                )
            )

        else:

            nearest_shot_lane_defender_offset_m = None


        auto_context_available = (
            nearest_defender_distance
            is not None
        )


        return {
            "shot_id":
                shot_id,

            "auto_context_available":
                auto_context_available,

            "detected_outfield_players":
                detected_outfield_players,

            "detected_defenders":
                detected_defenders,

            "detected_attackers":
                detected_attackers,

            "shooter_detection_id":
                shooter_detection_id,

            "shooter_distance_to_shot":
                shooter_distance_to_shot,

            "shooter_match_quality":
                shooter_match_quality,

            "nearest_defender_distance":
                nearest_defender_distance,

            "defenders_within_3m":
                defenders_within_3m,

            "defenders_within_5m":
                defenders_within_5m,

            "defenders_within_10m":
                defenders_within_10m,

            "pressure_score":
                pressure_score,

            "nearest_teammate_distance":
                nearest_teammate_distance,

            "teammates_within_5m":
                teammates_within_5m,

            "teammates_within_10m":
                teammates_within_10m,

            "goalkeeper_detection_id":
                goalkeeper_detection_id,

            "goalkeeper_pitch_x":
                goalkeeper_pitch_x,

            "goalkeeper_pitch_y":
                goalkeeper_pitch_y,

            "goalkeeper_depth_m":
                goalkeeper_depth_m,

            "goalkeeper_lateral_offset_m":
                goalkeeper_lateral_offset_m,

            "goalkeeper_shot_line_offset_m":
                goalkeeper_shot_line_offset_m,

            "defenders_between_shot_and_goal":
                defenders_between_shot_and_goal,

            "shot_lane_defenders_1m":
                shot_lane_defenders_1m,

            "shot_lane_defenders_2m":
                shot_lane_defenders_2m,

            "shot_lane_defenders_3m":
                shot_lane_defenders_3m,

            "nearest_shot_lane_defender_offset_m":
                nearest_shot_lane_defender_offset_m,
        }


    # ========================================================
    # xG INFERENCE
    # ========================================================

    def compute_xg_predictions(
        self,
        shot_bundle,
        context_result
    ):

        distance_to_goal = float(
            shot_bundle[
                "distance_to_goal"
            ]
        )


        shot_angle_degrees = float(
            shot_bundle[
                "shot_angle_degrees"
            ]
        )


        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------
        X_baseline = pd.DataFrame(
            [
                {
                    "distance_to_goal":
                        distance_to_goal,

                    "shot_angle":
                        shot_angle_degrees,
                }
            ]
        )


        X_baseline = X_baseline[
            BASELINE_FEATURES
        ]


        baseline_xg = float(
            self.baseline_model
            .predict_proba(
                X_baseline
            )[0, 1]
        )


        # ----------------------------------------------------
        # Enhanced automatic context
        # ----------------------------------------------------
        if (
            context_result[
                "auto_context_available"
            ]
            and
            context_result[
                "nearest_defender_distance"
            ] is not None
            and
            context_result[
                "defenders_within_5m"
            ] is not None
        ):

            X_enhanced = pd.DataFrame(
                [
                    {
                        "distance_to_goal":
                            distance_to_goal,

                        "shot_angle":
                            shot_angle_degrees,

                        "nearest_defender_distance":
                            float(
                                context_result[
                                    "nearest_defender_distance"
                                ]
                            ),

                        "defenders_within_5m":
                            float(
                                context_result[
                                    "defenders_within_5m"
                                ]
                            ),
                    }
                ]
            )


            X_enhanced = X_enhanced[
                ENHANCED_FEATURES
            ]


            automatic_context_xg = float(
                self.enhanced_model
                .predict_proba(
                    X_enhanced
                )[0, 1]
            )

        else:

            automatic_context_xg = None


        return {
            "shot_id":
                shot_bundle[
                    "shot_id"
                ],

            "baseline_xg":
                baseline_xg,

            "automatic_context_xg":
                automatic_context_xg,
        }


    # ========================================================
    # COMPLETE PIPELINE
    # ========================================================

    def analyze_shot(
        self,
        shot_id
    ) -> ShotAnalysisResult:

        bundle = self.get_shot_inputs(
            shot_id
        )


        projected = (
            self.project_and_filter_detections(
                bundle
            )
        )


        context = (
            self.compute_automatic_context(
                bundle,
                projected
            )
        )


        xg = (
            self.compute_xg_predictions(
                bundle,
                context
            )
        )


        if not bundle[
            "has_homography"
        ]:

            pipeline_status = (
                "BASELINE_ONLY_NO_HOMOGRAPHY"
            )

        elif not context[
            "auto_context_available"
        ]:

            pipeline_status = (
                "BASELINE_ONLY_NO_CONTEXT"
            )

        elif xg[
            "automatic_context_xg"
        ] is None:

            pipeline_status = (
                "CONTEXT_AVAILABLE_XG_FAILED"
            )

        else:

            pipeline_status = (
                "SUCCESS"
            )


        return ShotAnalysisResult(

            shot_id=
                shot_id,

            has_homography=
                bool(
                    bundle[
                        "has_homography"
                    ]
                ),

            has_automatic_context=
                bool(
                    context[
                        "auto_context_available"
                    ]
                ),

            shooting_team=
                bundle[
                    "shooting_team"
                ],

            defending_team=
                bundle[
                    "defending_team"
                ],

            shot_pitch_x=
                _safe_float(
                    bundle[
                        "shot_pitch_x"
                    ]
                ),

            shot_pitch_y=
                _safe_float(
                    bundle[
                        "shot_pitch_y"
                    ]
                ),

            distance_to_goal=
                _safe_float(
                    bundle[
                        "distance_to_goal"
                    ]
                ),

            shot_angle_radians=
                _safe_float(
                    bundle[
                        "shot_angle_radians"
                    ]
                ),

            shot_angle_degrees=
                _safe_float(
                    bundle[
                        "shot_angle_degrees"
                    ]
                ),

            detected_outfield_players=
                _safe_int(
                    context[
                        "detected_outfield_players"
                    ]
                ),

            detected_defenders=
                _safe_int(
                    context[
                        "detected_defenders"
                    ]
                ),

            detected_attackers=
                _safe_int(
                    context[
                        "detected_attackers"
                    ]
                ),

            shooter_detection_id=
                _safe_int(
                    context[
                        "shooter_detection_id"
                    ]
                ),

            shooter_distance_to_shot=
                _safe_float(
                    context[
                        "shooter_distance_to_shot"
                    ]
                ),

            shooter_match_quality=
                context[
                    "shooter_match_quality"
                ],

            nearest_defender_distance=
                _safe_float(
                    context[
                        "nearest_defender_distance"
                    ]
                ),

            defenders_within_3m=
                _safe_int(
                    context[
                        "defenders_within_3m"
                    ]
                ),

            defenders_within_5m=
                _safe_int(
                    context[
                        "defenders_within_5m"
                    ]
                ),

            defenders_within_10m=
                _safe_int(
                    context[
                        "defenders_within_10m"
                    ]
                ),

            pressure_score=
                _safe_float(
                    context[
                        "pressure_score"
                    ]
                ),

            nearest_teammate_distance=
                _safe_float(
                    context[
                        "nearest_teammate_distance"
                    ]
                ),

            teammates_within_5m=
                _safe_int(
                    context[
                        "teammates_within_5m"
                    ]
                ),

            teammates_within_10m=
                _safe_int(
                    context[
                        "teammates_within_10m"
                    ]
                ),

            goalkeeper_detection_id=
                _safe_int(
                    context[
                        "goalkeeper_detection_id"
                    ]
                ),

            goalkeeper_pitch_x=
                _safe_float(
                    context[
                        "goalkeeper_pitch_x"
                    ]
                ),

            goalkeeper_pitch_y=
                _safe_float(
                    context[
                        "goalkeeper_pitch_y"
                    ]
                ),

            goalkeeper_depth_m=
                _safe_float(
                    context[
                        "goalkeeper_depth_m"
                    ]
                ),

            goalkeeper_lateral_offset_m=
                _safe_float(
                    context[
                        "goalkeeper_lateral_offset_m"
                    ]
                ),

            goalkeeper_shot_line_offset_m=
                _safe_float(
                    context[
                        "goalkeeper_shot_line_offset_m"
                    ]
                ),

            defenders_between_shot_and_goal=
                _safe_int(
                    context[
                        "defenders_between_shot_and_goal"
                    ]
                ),

            shot_lane_defenders_1m=
                _safe_int(
                    context[
                        "shot_lane_defenders_1m"
                    ]
                ),

            shot_lane_defenders_2m=
                _safe_int(
                    context[
                        "shot_lane_defenders_2m"
                    ]
                ),

            shot_lane_defenders_3m=
                _safe_int(
                    context[
                        "shot_lane_defenders_3m"
                    ]
                ),

            baseline_xg=
                _safe_float(
                    xg[
                        "baseline_xg"
                    ]
                ),

            automatic_context_xg=
                _safe_float(
                    xg[
                        "automatic_context_xg"
                    ]
                ),

            pipeline_status=
                pipeline_status,
        )


# ============================================================
# DEFAULT PROJECT ROOT + CONVENIENCE FUNCTION
# ============================================================

DEFAULT_PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


_DEFAULT_PIPELINE = None


def get_default_pipeline(
    project_root=None
):

    global _DEFAULT_PIPELINE


    root = (
        Path(project_root).resolve()
        if project_root is not None
        else DEFAULT_PROJECT_ROOT
    )


    if (
        _DEFAULT_PIPELINE is None
        or
        _DEFAULT_PIPELINE.project_root != root
    ):

        _DEFAULT_PIPELINE = (
            SoccerShotPipeline(
                root
            )
        )


    return _DEFAULT_PIPELINE


def analyze_shot(
    shot_id,
    project_root=None
):
    """
    Convenience function for one-shot analysis.

    Example
    -------
    result = analyze_shot(
        "crystal_palace_arsenal_h1_shot_03"
    )
    """

    pipeline = get_default_pipeline(
        project_root=project_root
    )

    return pipeline.analyze_shot(
        shot_id
    )


__all__ = [
    "ShotAnalysisResult",
    "SoccerShotPipeline",
    "get_default_pipeline",
    "analyze_shot",
]

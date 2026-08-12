import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import json
import csv
import random
from datetime import datetime
from collections import deque

try:
    import pygame
except Exception:
    pygame = None

# -----------------------------
# Game files
# -----------------------------
BACKGROUND_PATH = "background_game.png"
SUN_SHEET_PATH = "sun_sheet.png"
CLOUD_SHEET_PATH = "kawaii_cloud_faces_grid_pattern.png"

# -----------------------------
# Data / profile / music files
# -----------------------------
DATA_DIR = "bloom_motion_data"
PROFILES_PATH = os.path.join(DATA_DIR, "profiles.json")
ACTIVE_PROFILE_PATH = os.path.join(DATA_DIR, "active_profile.json")
SESSION_PROGRESS_CSV_PATH = os.path.join(DATA_DIR, "session_progress.csv")
SESSION_ANALYSIS_CSV_PATH = os.path.join(DATA_DIR, "session_angle_samples.csv")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

MUSIC_DIR = "music"
SUPPORTED_MUSIC_EXTENSIONS = (".mp3", ".wav", ".ogg")

# -----------------------------
# Analysis / ROM chart settings
# -----------------------------
# The Analysis screen uses the old Progress history plus new per-session
# angle samples for Flexion, Extension, Left Side Bend and Right Side Bend.
ANALYSIS_ROM_MOVEMENTS = [
    ("flexion", "Flexion"),
    ("extension", "Extension"),
    ("left_bend", "Left Side Bend"),
    ("right_bend", "Right Side Bend"),
]

# The dropdown shows every calibrated/game movement, but the angle-time
# chart is available only for the four ROM angle movements above.
ANALYSIS_ALL_MOVEMENTS = [
    ("flexion", "Flexion"),
    ("extension", "Extension"),
    ("left_bend", "Left Side Bend"),
    ("right_bend", "Right Side Bend"),
    ("chin_tuck", "Chin Tuck"),
    ("shoulder_lift", "Scapular Elevation"),
    ("palm_retraction", "Scapular Retraction"),
]

ANALYSIS_ROM_MOVEMENT_KEYS = {item[0] for item in ANALYSIS_ROM_MOVEMENTS}
ANALYSIS_ALL_MOVEMENT_KEYS = {item[0] for item in ANALYSIS_ALL_MOVEMENTS}
ANALYSIS_SAMPLE_INTERVAL_SECONDS = 0.20
ANALYSIS_MIN_ACTIVE_ANGLE_DEG = 2.0
ANALYSIS_ERROR_TOLERANCE_DEG = 10.0
ANALYSIS_GOOD_ACCURACY_PERCENT = 75.0
ANALYSIS_GOOD_ERROR_DEG = 6.0
ANALYSIS_MEDIUM_ERROR_DEG = 12.0

# The Analysis dashboard should be supportive and time-based.
# A sample is taken about every ANALYSIS_SAMPLE_INTERVAL_SECONDS, so quality
# is calculated from active movement time instead of punishing every tiny
# frame-level mismatch as a hard error.
ANALYSIS_MIN_ACTIVE_TIME_SECONDS = 2.0
ANALYSIS_GOOD_QUALITY_PERCENT = 75.0
ANALYSIS_ACCEPTABLE_QUALITY_PERCENT = 50.0
ANALYSIS_ACTIVE_SAMPLE_TIME_FALLBACK = ANALYSIS_SAMPLE_INTERVAL_SECONDS

# Optional menu design images.
# Put these image files next to this Python file if you want the menu to use them.
# If they do not exist, the code automatically uses the drawn fallback.
MAIN_MENU_BACKGROUND_PATH = "main_menu_background.png"
MENU_PANEL_BACKGROUND_PATH = "menu_panel_background.png"

# Stage selection background. The user can save the background image as
# backlevel.png / backlevel.jpg / backlevel.jpeg / backlevel.webp next to this file.
BACKLEVEL_BACKGROUND_PATHS = [
    "backlevel.png",
    "backlevel.jpg",
    "backlevel.jpeg",
    "backlevel.webp",
    "backlevel",
]

# Optional stage selector preview images.
# Save the new images next to this Python file with these exact names.
STAGE2_PREVIEW_PATH = "stage2_flower_patch.png"
STAGE3_PREVIEW_PATH = "stage3_flower_trellis.png"
STAGE4_PREVIEW_PATH = "back4.png"

# Easy tutorial stage assets.
# Put these files next to this Python file:
# back1.png, tree_stage1.png, tree_stage2.png, tree_stage3.png
TUTORIAL_BACKGROUND_PATH = "back1.png"
TREE_STAGE1_PATH = "tree_stage1.png"
TREE_STAGE2_PATH = "tree_stage2.png"
TREE_STAGE3_PATH = "tree_stage3.png"

# Stage 2 - Summer Garden assets.
# Save the summer background as back2.png.
# Save the two bush sets with these exact names next to this Python file.
STAGE2_BACKGROUND_PATH = "back2.png"
STAGE2_LEFT_BUSH_STAGE1_PATH = "flower_bush_stage1.png"
STAGE2_LEFT_BUSH_STAGE2_PATH = "flower_bush_stage2.png"
STAGE2_LEFT_BUSH_STAGE3_PATH = "flower_bush_stage3.png"
STAGE2_RIGHT_BUSH_STAGE1_PATH = "purple_bush_stage1.png"
STAGE2_RIGHT_BUSH_STAGE2_PATH = "purple_bush_stage2.png"
STAGE2_RIGHT_BUSH_STAGE3_PATH = "purple_bush_stage3.png"

# Stage 3 - Autumn Garden assets.
# Save these files next to this Python file with these exact names.
STAGE3_BACKGROUND_PATH = "back3.png"
STAGE3_CHRYSANTHEMUM_STAGE1_PATH = "stage3_chrysanthemum_stage1.png"
STAGE3_CHRYSANTHEMUM_STAGE2_PATH = "stage3_chrysanthemum_stage2.png"
STAGE3_CHRYSANTHEMUM_STAGE3_PATH = "stage3_chrysanthemum_stage3.png"
STAGE3_MAPLE_STAGE1_PATH = "stage3_maple_stage1.png"
STAGE3_MAPLE_STAGE2_PATH = "stage3_maple_stage2.png"
STAGE3_MAPLE_STAGE3_PATH = "stage3_maple_stage3.png"
STAGE3_PURPLE_BUSH_STAGE1_PATH = "stage3_purple_bush_stage1.png"
STAGE3_PURPLE_BUSH_STAGE2_PATH = "stage3_purple_bush_stage2.png"
STAGE3_PURPLE_BUSH_STAGE3_PATH = "stage3_purple_bush_stage3.png"

# Stage 4 - Winter Garden assets.
# Save the winter background as back4.png and the twelve plant images
# next to this Python file using these exact names.
STAGE4_BACKGROUND_PATH = "back4.png"
STAGE4_WINTER_ROSE_STAGE1_PATH = "stage4_winter_rose_stage1.png"
STAGE4_WINTER_ROSE_STAGE2_PATH = "stage4_winter_rose_stage2.png"
STAGE4_WINTER_ROSE_STAGE3_PATH = "stage4_winter_rose_stage3.png"
STAGE4_SNOWDROP_STAGE1_PATH = "stage4_snowdrop_stage1.png"
STAGE4_SNOWDROP_STAGE2_PATH = "stage4_snowdrop_stage2.png"
STAGE4_SNOWDROP_STAGE3_PATH = "stage4_snowdrop_stage3.png"
STAGE4_POINSETTIA_STAGE1_PATH = "stage4_poinsettia_stage1.png"
STAGE4_POINSETTIA_STAGE2_PATH = "stage4_poinsettia_stage2.png"
STAGE4_POINSETTIA_STAGE3_PATH = "stage4_poinsettia_stage3.png"
STAGE4_CYCLAMEN_STAGE1_PATH = "stage4_cyclamen_stage1.png"
STAGE4_CYCLAMEN_STAGE2_PATH = "stage4_cyclamen_stage2.png"
STAGE4_CYCLAMEN_STAGE3_PATH = "stage4_cyclamen_stage3.png"


TOP_FLOWER_STAGE1_PATH = "flower_stage1.png"
TOP_FLOWER_STAGE2_PATH = "flower_stage2.png"
TOP_FLOWER_STAGE3_PATH = "flower_stage3.png"

BOTTOM_FLOWER_STAGE1_PATH = "rose_stage1.png"
BOTTOM_FLOWER_STAGE2_PATH = "rose_stage2.png"
BOTTOM_FLOWER_STAGE3_PATH = "rose_stage3.png"

ORCHID_STAGE1_PATH = "orchid_stage1.png"
ORCHID_STAGE2_PATH = "orchid_stage2.png"
ORCHID_STAGE3_PATH = "orchid_stage3.png"

TULIP_STAGE1_PATH = "tulip_stage1.png"
TULIP_STAGE2_PATH = "tulip_stage2.png"
TULIP_STAGE3_PATH = "tulip_stage3.png"

BLUEBLOOM_STAGE1_PATH = "bluebloom_stage1.png"
BLUEBLOOM_STAGE2_PATH = "bluebloom_stage2.png"
BLUEBLOOM_STAGE3_PATH = "bluebloom_stage3.png"

PEONY_STAGE1_PATH = "peony_stage1.png"
PEONY_STAGE2_PATH = "peony_stage2.png"
PEONY_STAGE3_PATH = "peony_stage3.png"

# -----------------------------
# Window settings
# -----------------------------
WIDTH = 1280
HEIGHT = 720
CAMERA_INDEX = 0

# -----------------------------
# Easy tutorial stage settings
# -----------------------------
# The tutorial is shown after full calibration and before the main six-pot game.
TUTORIAL_TREE_CENTER_X = 510
TUTORIAL_TREE_CENTER_Y = 340
TUTORIAL_TREE_BASE_Y = 407
TUTORIAL_TARGET_RADIUS = 82

# Move only the drawn tree a little to the right while keeping the target circle stable.
TUTORIAL_TREE_DRAW_OFFSET_X = 24

# After the tree reaches Stage 3, keep it visible for a few seconds before showing the win menu.
TUTORIAL_STAGE3_TO_WIN_DELAY = 4.0

TUTORIAL_SUN_START_X = WIDTH - 245
TUTORIAL_SUN_START_Y = HEIGHT - 205
TUTORIAL_SUN_LOCK_X = TUTORIAL_TREE_CENTER_X - SUN_SIZE // 2 if "SUN_SIZE" in globals() else WIDTH // 2 - 65
TUTORIAL_SUN_LOCK_Y = 145

TUTORIAL_MOVE_DISTANCE = 58
TUTORIAL_CHIN_REQUIRED_TOTAL_TIME = 10.0

# -----------------------------
# Stage 2 - Summer Garden settings
# -----------------------------
# This stage is intentionally separate from Stage 1 and Stage 5 so the
# existing tutorial and six-pot main garden logic stay safe.
STAGE2_TOTAL_POTS = 2
STAGE2_BACKGROUND_OVERLAY_ALPHA = 0.0

# Character starts near the lower path. Values are top-left coordinates of the sun/cloud.
STAGE2_SUN_START_X = 575
STAGE2_SUN_START_Y = 545
STAGE2_MOVE_DISTANCE = 82
STAGE2_MOVE_COOLDOWN = 0.35
STAGE2_VISUAL_SMOOTHING_FACTOR = 0.38
# Free movement across the full visible Stage 2 window.
STAGE2_SUN_MIN_X = 0
STAGE2_SUN_MAX_X = WIDTH - 130
STAGE2_SUN_MIN_Y = 0
STAGE2_SUN_MAX_Y = HEIGHT - 130

# Two summer pots. pot_soil_y is the visual soil line where the bush grows from.
STAGE2_LEFT_POT_CENTER_X = 285
STAGE2_LEFT_POT_SOIL_Y = 290
STAGE2_RIGHT_POT_CENTER_X = 1000
STAGE2_RIGHT_POT_SOIL_Y = 290
STAGE2_LOCK_GAP_ABOVE_POT = 84

# Road rectangles are based on the character center, not the top-left sprite corner.
# The rectangles cover the clean dirt paths of back2.png and avoid the grass/pond.
STAGE2_DIRT_ROAD_RECTS = [
    (470, 430, 810, 705),   # lower vertical approach
    (470, 175, 810, 455),   # middle/top approach around the pond
    (170, 185, 475, 465),   # left pot road loop
    (805, 185, 1110, 465),  # right pot road loop
    (285, 135, 995, 285),   # upper bridge road
]

STAGE2_LEFT_TRIGGER_RECT = (190, 210, 415, 445)
STAGE2_RIGHT_TRIGGER_RECT = (865, 210, 1090, 445)

STAGE2_CHIN_REQUIRED_TOTAL_TIME = 10.0
STAGE2_SHOULDER_REQUIRED_HOLD_TIME = 5.0
STAGE2_RETRACTION_REQUIRED_HOLD_TIME = 10.0
STAGE2_RAIN_DURATION = 2.0
STAGE2_RETURN_DELAY_AFTER_POT = 1.5
STAGE2_WIN_DELAY = 4.0
STAGE2_BUSH_SIZE = 145

# -----------------------------
# Stage 3 - Autumn Garden settings
# -----------------------------
STAGE3_TOTAL_POTS = 3

# Top-left coordinate of the sun. Its center starts directly in front of
# the stone entrance at the top of back3.png.
STAGE3_SUN_START_X = 435
STAGE3_SUN_START_Y = 80
STAGE3_MOVE_DISTANCE = 68
STAGE3_MOVE_COOLDOWN = 0.38
STAGE3_VISUAL_SMOOTHING_FACTOR = 0.38
# Free movement across the full visible Stage 3 window.
STAGE3_SUN_MIN_X = 0
STAGE3_SUN_MAX_X = WIDTH - 130
STAGE3_SUN_MIN_Y = 0
STAGE3_SUN_MAX_Y = HEIGHT - 130

# Plant soil positions on back3.png after it is resized to 1280x720.
STAGE3_CHRYSANTHEMUM_CENTER_X = 300
STAGE3_CHRYSANTHEMUM_SOIL_Y = 455
STAGE3_MAPLE_CENTER_X = 825
STAGE3_MAPLE_SOIL_Y = 210
STAGE3_PURPLE_BUSH_CENTER_X = 998
STAGE3_PURPLE_BUSH_SOIL_Y = 500
STAGE3_LOCK_GAP_ABOVE_POT = 78
STAGE3_MAPLE_LOCK_GAP_ABOVE_POT = 20

# Wide walkable rectangles based on the center of the sun/cloud.
# Together they follow the broad stone paths in back3.png.
STAGE3_DIRT_ROAD_RECTS = [
    (450, 105, 545, 535),    # entrance and main vertical path
    (170, 245, 1160, 390),   # broad central horizontal path
    (135, 335, 370, 535),    # lower-left plant approach
    (780, 240, 1160, 550),   # right-side and lower-right paths
    (780, 155, 875, 300),    # upper-right maple approach
    (170, 455, 1160, 535),   # lower connecting path
]

STAGE3_CHRYSANTHEMUM_TRIGGER_RECT = (225, 395, 355, 510)
STAGE3_MAPLE_TRIGGER_RECT = (770, 170, 880, 260)
STAGE3_PURPLE_BUSH_TRIGGER_RECT = (925, 430, 1060, 550)

STAGE3_CHIN_REQUIRED_TOTAL_TIME = 10.0
STAGE3_SHOULDER_REQUIRED_HOLD_TIME = 5.0
STAGE3_RETRACTION_REQUIRED_HOLD_TIME = 10.0
STAGE3_RAIN_DURATION = 2.0
STAGE3_RETURN_DELAY_AFTER_POT = 1.5
STAGE3_WIN_DELAY = 4.0
STAGE3_PLANT_SIZE = 176
STAGE3_MAPLE_SIZE = 215

# -----------------------------
# Stage 4 - Winter Garden settings
# -----------------------------
STAGE4_TOTAL_POTS = 4

# The sun starts just inside the lower entrance gate of back4.png.
STAGE4_SUN_START_X = 500
STAGE4_SUN_START_Y = 535
STAGE4_MOVE_DISTANCE = 68
STAGE4_MOVE_COOLDOWN = 0.38
STAGE4_VISUAL_SMOOTHING_FACTOR = 0.38
# Free movement across the full visible Stage 4 window.
STAGE4_SUN_MIN_X = 0
STAGE4_SUN_MAX_X = WIDTH - 130
STAGE4_SUN_MIN_Y = 0
STAGE4_SUN_MAX_Y = HEIGHT - 130

# Plant soil positions after back4.png is resized to 1280 x 720.
STAGE4_WINTER_ROSE_CENTER_X = 365
STAGE4_WINTER_ROSE_SOIL_Y = 305
STAGE4_SNOWDROP_CENTER_X = 820
STAGE4_SNOWDROP_SOIL_Y = 235
STAGE4_POINSETTIA_CENTER_X = 300
STAGE4_POINSETTIA_SOIL_Y = 478
STAGE4_CYCLAMEN_CENTER_X = 990
STAGE4_CYCLAMEN_SOIL_Y = 520

# The upper-right pot needs a smaller vertical gap so the character stays on-screen.
STAGE4_LOCK_GAP_ABOVE_POT = 58
STAGE4_SNOWDROP_LOCK_GAP_ABOVE_POT = 12

# Wide walkable rectangles follow the pale stone paths in back4.png.
# Coordinates are based on the center of the sun/cloud.
STAGE4_DIRT_ROAD_RECTS = [
    (430, 95, 585, 390),     # upper entrance and center vertical path
    (430, 235, 900, 330),    # upper horizontal path
    (135, 320, 590, 405),    # left upper branch
    (120, 325, 245, 555),    # far-left vertical loop
    (175, 430, 930, 535),    # lower horizontal path
    (760, 230, 930, 545),    # middle-right vertical path
    (760, 330, 1190, 415),   # right upper branch
    (1070, 330, 1190, 565),  # far-right vertical loop
    (430, 470, 610, 650),    # lower entrance connector
    (250, 180, 585, 345),    # upper-left pot connector
    (850, 390, 1120, 560),   # lower-right pot connector
]

STAGE4_WINTER_ROSE_TRIGGER_RECT = (300, 225, 430, 325)
STAGE4_SNOWDROP_TRIGGER_RECT = (755, 155, 885, 280)
STAGE4_POINSETTIA_TRIGGER_RECT = (235, 395, 365, 510)
STAGE4_CYCLAMEN_TRIGGER_RECT = (920, 430, 1060, 555)

STAGE4_CHIN_REQUIRED_TOTAL_TIME = 10.0
STAGE4_SHOULDER_REQUIRED_HOLD_TIME = 5.0
STAGE4_RETRACTION_REQUIRED_HOLD_TIME = 10.0
STAGE4_RAIN_DURATION = 2.0
STAGE4_RETURN_DELAY_AFTER_POT = 1.5
STAGE4_WIN_DELAY = 4.0
STAGE4_PLANT_SIZE = 178

# -----------------------------
# Sun / Cloud settings
# -----------------------------
SUN_SIZE = 130

# Start point of the character on the dirt road
sun_x = 575
sun_y = 300

# Dirt-road center point.
# Vertical movement is allowed only around this X.
# Horizontal movement is allowed only around this Y.
ROAD_CENTER_X = sun_x
ROAD_CENTER_Y = sun_y

sun_current_x = float(sun_x)
sun_current_y = float(sun_y)

sun_target_x = float(sun_x)
sun_target_y = float(sun_y)

SUN_MOVE_DISTANCE = 60

# Stage 5 free-movement limits.
# The character may move anywhere inside the visible game window without leaving the screen.
SUN_MIN_Y = 0
SUN_MAX_Y = HEIGHT - SUN_SIZE

SUN_MIN_X = 0
SUN_MAX_X = WIDTH - SUN_SIZE

# Character cannot leave the dirt-road cross path.
ROAD_LOCK_TOLERANCE = 35

SUN_MOVE_COOLDOWN = 0.8
last_sun_move_time = 0

SUN_SHINING_DURATION = 2.0
sun_shining_start_time = 0

active_character = "sun"  # "sun" or "cloud"

# Easy tutorial stage runtime variables.
tutorial_sun_current_x = float(TUTORIAL_SUN_START_X)
tutorial_sun_current_y = float(TUTORIAL_SUN_START_Y)
tutorial_sun_target_x = float(TUTORIAL_SUN_START_X)
tutorial_sun_target_y = float(TUTORIAL_SUN_START_Y)

tutorial_locked_to_center = False
tutorial_chin_tuck_total_time = 0.0
tutorial_chin_tuck_last_update_time = None
tutorial_message = "Move the sun to the center tree circle."
tutorial_completed = False
tutorial_stage3_complete_time = None

# Stage 2 runtime variables.
stage2_sun_current_x = float(STAGE2_SUN_START_X)
stage2_sun_current_y = float(STAGE2_SUN_START_Y)
stage2_sun_target_x = float(STAGE2_SUN_START_X)
stage2_sun_target_y = float(STAGE2_SUN_START_Y)

stage2_left_bush_stage = 0
stage2_right_bush_stage = 0
stage2_score = 0

stage2_locked_to_pot = False
stage2_locked_pot_key = None
stage2_active_pot_key = None
stage2_message = "Move the sun to one of the empty summer pots."
stage2_completed = False
stage2_completion_time = None

stage2_chin_tuck_total_time = 0.0
stage2_chin_tuck_last_update_time = None
stage2_shoulder_hold_start = None
stage2_shoulder_release_start_time = None
stage2_shoulder_total_time = 0.0
stage2_shoulder_last_update_time = None
stage2_retraction_hold_start = None
stage2_retraction_last_seen_time = None
stage2_retraction_total_time = 0.0
stage2_retraction_last_update_time = None

stage2_rain_sequence_active = False
stage2_rain_pot_key = None
stage2_rain_start_time = 0.0
stage2_stage3_pause_active = False
stage2_stage3_pause_pot_key = None
stage2_stage3_pause_start_time = 0.0

# Stage 3 runtime variables. Kept separate so Stage 2 and Stage 5 cannot
# accidentally share plant stages, locks, or timers.
stage3_sun_current_x = float(STAGE3_SUN_START_X)
stage3_sun_current_y = float(STAGE3_SUN_START_Y)
stage3_sun_target_x = float(STAGE3_SUN_START_X)
stage3_sun_target_y = float(STAGE3_SUN_START_Y)

stage3_chrysanthemum_stage = 0
stage3_maple_stage = 0
stage3_purple_bush_stage = 0
stage3_score = 0

stage3_locked_to_pot = False
stage3_locked_pot_key = None
stage3_active_pot_key = None
stage3_message = "Move the sun from the entrance to one of the autumn pots."
stage3_completed = False
stage3_completion_time = None

stage3_chin_tuck_total_time = 0.0
stage3_chin_tuck_last_update_time = None
stage3_shoulder_hold_start = None
stage3_shoulder_release_start_time = None
stage3_shoulder_total_time = 0.0
stage3_shoulder_last_update_time = None
stage3_retraction_hold_start = None
stage3_retraction_last_seen_time = None
stage3_retraction_total_time = 0.0
stage3_retraction_last_update_time = None

stage3_rain_sequence_active = False
stage3_rain_pot_key = None
stage3_rain_start_time = 0.0
stage3_stage3_pause_active = False
stage3_stage3_pause_pot_key = None
stage3_stage3_pause_start_time = 0.0

# Stage 4 runtime variables. These remain separate from every other stage.
stage4_sun_current_x = float(STAGE4_SUN_START_X)
stage4_sun_current_y = float(STAGE4_SUN_START_Y)
stage4_sun_target_x = float(STAGE4_SUN_START_X)
stage4_sun_target_y = float(STAGE4_SUN_START_Y)

stage4_winter_rose_stage = 0
stage4_snowdrop_stage = 0
stage4_poinsettia_stage = 0
stage4_cyclamen_stage = 0
stage4_score = 0

stage4_locked_to_pot = False
stage4_locked_pot_key = None
stage4_active_pot_key = None
stage4_message = "Move the sun from the lower gate to one of the winter pots."
stage4_completed = False
stage4_completion_time = None

stage4_chin_tuck_total_time = 0.0
stage4_chin_tuck_last_update_time = None
stage4_shoulder_hold_start = None
stage4_shoulder_release_start_time = None
stage4_shoulder_total_time = 0.0
stage4_shoulder_last_update_time = None
stage4_retraction_hold_start = None
stage4_retraction_last_seen_time = None
stage4_retraction_total_time = 0.0
stage4_retraction_last_update_time = None

stage4_rain_sequence_active = False
stage4_rain_pot_key = None
stage4_rain_start_time = 0.0
stage4_stage3_pause_active = False
stage4_stage3_pause_pot_key = None
stage4_stage3_pause_start_time = 0.0

# -----------------------------
# Mouse / screen interaction settings
# -----------------------------
mouse_x = -1
mouse_y = -1
mouse_left_clicked = False
quit_game = False
win_message = ""

# -----------------------------
# Stage selection / level system
# -----------------------------
# Stage 1 is the current easy one-tree tutorial.
# Stage 5 is the current six-pot main garden.
# Temporary development switch: while this is True, every stage card is shown
# without a lock. Implemented stages can be played immediately, while stages
# that are not implemented yet remain visible as Coming Soon.
TEMP_UNLOCK_ALL_STAGES = True

selected_stage_number = None
current_stage_number = None
stage_select_message = ""

# Home / pause menu interaction
HOME_ICON_BUTTON_RECT = (1188, 24, 1248, 84)
pause_menu_enter_time = None
pause_return_state = "game"  # "game", "tutorial", "stage2", "stage3", or "stage4"
calibration_return_mode = "new_game"  # "new_game" or "resume_game"

# Recalibration selection from the Home/Pause menu.
# None = normal full calibration flow. Otherwise it stores the selected movement id.
selected_recalibration_target = None


# -----------------------------
# Rain settings
# -----------------------------
RAIN_EFFECT_DURATION = 2.0
rain_effect_start_time = 0
rain_effect_x = float(sun_x)
rain_effect_y = float(sun_y)

# بعد از اینکه حرکت عقب بردن شانه/کتف ۳ ثانیه کامل شد:
# 1) ابر همان بالا می‌ماند و ۲ ثانیه باران می‌بارد.
# 2) بعد از ۲ ثانیه، گل Stage 3 می‌شود.
# 3) بعد از ۱.۵ ثانیه مکث، کاراکتر به خورشید تبدیل می‌شود و به مرکز بازی برمی‌گردد.
POST_STAGE3_RETURN_DELAY = 1.5

locked_rain_sequence_active = False
locked_rain_flower_key = None
locked_rain_start_time = 0.0

locked_stage3_pause_active = False
locked_stage3_pause_flower_key = None
locked_stage3_pause_start_time = 0.0

# -----------------------------
# Flower / Score settings
# -----------------------------
score = 0

TOTAL_FLOWERS = 6
game_finished = False

top_flower_stage = 0
bottom_flower_stage = 0

right_orchid_stage = 0
south_east_bluebloom_stage = 0
left_tulip_stage = 0
south_west_peony_stage = 0

FLOWER_SIZE = 120

top_flower_animating = False
bottom_flower_animating = False

right_orchid_animating = False
south_east_bluebloom_animating = False
left_tulip_animating = False
south_west_peony_animating = False

top_flower_start_time = 0
bottom_flower_start_time = 0

right_orchid_start_time = 0
south_east_bluebloom_start_time = 0
left_tulip_start_time = 0
south_west_peony_start_time = 0

TOP_POT_CENTER_X = 625
TOP_POT_SOIL_Y = 220

BOTTOM_POT_CENTER_X = 625
BOTTOM_POT_SOIL_Y = 610

# Side pot flower positions
# If a flower is not exactly centered on its pot, only adjust these numbers.
RIGHT_ORCHID_POT_CENTER_X = 1150
RIGHT_ORCHID_POT_SOIL_Y = 345

# The former lower-right orange pot has moved to the upper-right branch.
# These coordinates match the updated 1280 x 720 Stage 5 background.
SOUTH_EAST_BLUEBLOOM_POT_CENTER_X = 1059
SOUTH_EAST_BLUEBLOOM_POT_SOIL_Y = 194

LEFT_TULIP_POT_CENTER_X = 145
LEFT_TULIP_POT_SOIL_Y = 345

# The former lower-left pink pot has moved to the upper-left branch.
SOUTH_WEST_PEONY_POT_CENTER_X = 213
SOUTH_WEST_PEONY_POT_SOIL_Y = 190

FLOWER_POT_OVERLAP = 0

# Legacy values are kept for compatibility, but Stage 5 now activates every pot
# using one proximity check, so the character may approach from any direction.
TOP_POT_TRIGGER_Y = 95
BOTTOM_POT_TRIGGER_Y = 500
STAGE5_POT_TRIGGER_HALF_WIDTH = 105
STAGE5_POT_TRIGGER_HALF_HEIGHT = 80
STAGE5_POT_TRIGGER_CENTER_Y_OFFSET = -55

active_flower = None  # None / "top" / "bottom" / "right_orchid" / "south_east_bluebloom" / "left_tulip" / "south_west_peony"

# -----------------------------
# Flower lock settings - Step 1
# -----------------------------
# وقتی خورشید به یک گلدان می‌رسد، دیگر به مرکز برنمی‌گردد.
# به جای آن، بالای همان گلدان قرار می‌گیرد و همان‌جا قفل می‌شود.
character_locked_to_flower = False
locked_flower_key = None

# فاصله عمودی خورشید از خاک گلدان وقتی بالای گل قفل می‌شود.
# اگر خورشید خیلی بالا/پایین بود، فقط این عدد را تغییر بده.
LOCK_GAP_ABOVE_POT = 80

# -----------------------------
# Locked flower Chin Tuck settings - Step 2
# -----------------------------
# در حالت قفل، فقط Chin Tuck مجاز است.
# این زمان، جدا از CHIN_REQUIRED_HOLD_TIME قدیمی است تا منطق قبلی باران خراب نشود.
LOCKED_CHIN_REQUIRED_TOTAL_TIME = 10.0

# شمارش Chin Tuck در حالت قفل تجمعی است.
# یعنی اگر کاربر ۲ ثانیه درست انجام داد، قطع شد، و بعد ادامه داد، از همان ۲ ثانیه ادامه می‌دهد.
locked_chin_tuck_total_time = 0.0
locked_chin_tuck_last_update_time = None

# -----------------------------
# Locked flower Scapular Elevation settings - Step 3
# -----------------------------
# بعد از Stage 2، در حالت قفل فقط بالا بردن شانه مجاز است.
# اگر شانه‌ها ۵ ثانیه بالا بمانند، خورشید همان‌جا به ابر تبدیل می‌شود.
LOCKED_SHOULDER_REQUIRED_HOLD_TIME = 5.0
locked_shoulder_hold_start = None
locked_shoulder_release_start_time = None
locked_shoulder_total_time = 0.0
locked_shoulder_last_update_time = None

# -----------------------------
# Side pots temporary settings
# -----------------------------
# For now, these only detect that the sun/cloud has reached side pots.
# Flower logic for these side pots will be added in the next step.
LEFT_POT_1_TRIGGER_X = 420
LEFT_POT_2_TRIGGER_X = 250
RIGHT_POT_1_TRIGGER_X = 730
RIGHT_POT_2_TRIGGER_X = 900

reached_side_pots = set()

SIDE_POTS = [
    ("left_1", "Left pot 1", "left", LEFT_POT_1_TRIGGER_X),
    ("left_2", "Left pot 2", "left", LEFT_POT_2_TRIGGER_X),
    ("right_1", "Right pot 1", "right", RIGHT_POT_1_TRIGGER_X),
    ("right_2", "Right pot 2", "right", RIGHT_POT_2_TRIGGER_X),
]

# -----------------------------
# Flexion / Extension settings
# -----------------------------
SMOOTHING = 0.85

FLEXION_REQUIRED_HOLD_TIME = 0.8
EXTENSION_REQUIRED_HOLD_TIME = 0.45

MIN_FLEXION_THRESHOLD = 7.0
MIN_EXTENSION_THRESHOLD = 4.5

FLEXION_THRESHOLD_RATIO = 0.60
EXTENSION_THRESHOLD_RATIO = 0.45

MAX_ALLOWED_YAW_CHANGE = 12.0

MIN_FLEXION_SAMPLE_DELTA = 5.0
MIN_EXTENSION_SAMPLE_DELTA = 3.5

# -----------------------------
# Left / Right Side Bend settings
# -----------------------------
# This is lateral neck flexion: bending the head toward the left/right shoulder.
# Detection uses the eye-line angle, which already exists in get_eye_roll().
SIDE_BEND_REQUIRED_HOLD_TIME = 0.45

MIN_SIDE_BEND_THRESHOLD = 4.0
SIDE_BEND_THRESHOLD_RATIO = 0.45

MIN_SIDE_BEND_SAMPLE_DELTA = 4.0

# Prevent side-bend from being confused with flexion/extension or face rotation.
MAX_ALLOWED_PITCH_CHANGE_FOR_SIDE_BEND = 999.0
MAX_ALLOWED_YAW_CHANGE_FOR_SIDE_BEND = 999.0

# -----------------------------
# Chin Tuck settings
# -----------------------------
CHIN_REQUIRED_HOLD_TIME = 3.0

# تشخیص Chin Tuck فقط برای مرحله خورشید/گل استفاده می‌شود؛ در حالت ابر غیرفعال است.
CHIN_MATCH_THRESHOLD = 1.35
CHIN_MIN_TARGET_STRENGTH = 0.004
CHIN_ENOUGH_MOVEMENT_RATIO = 0.20

CHIN_PROGRESS_MIN = 0.25
CHIN_PROGRESS_MAX = 4.50
CHIN_SIDE_ERROR_MAX = 1.85

# اگر اینها خیلی سخت‌گیر باشند، Chin Tuck از دست می‌رود.
# فعلاً متوسط هستند تا Flexion/Extension با Chin Tuck قاطی نشود.
CHIN_HEAD_PITCH_LIMIT = 13.0
CHIN_HEAD_YAW_LIMIT = 13.0
CHIN_HEAD_ROLL_LIMIT = 15.0

CHIN_ANGLE_SMOOTHING = 0.85

CHIN_NEUTRAL_AVERAGE_FRAMES = 20
CHIN_TARGET_AVERAGE_FRAMES = 8
CHIN_CURRENT_AVERAGE_FRAMES = 5

# اگر Chin Tuck برای لحظه کوتاه به خاطر نویز گم شد، تایمر قطع نشود.
CHIN_MISS_TOLERANCE_TIME = 0.45

# بعد از تبدیل خورشید به ابر، باران همان لحظه خودکار شروع نشود.
RAIN_AFTER_CLOUD_COOLDOWN = 0.25

# برای باران، شانه‌ها باید کمی از حالت بالا آمده برگردند، اما خیلی سخت نمی‌گیریم.
RAIN_SHOULDER_RETURN_TO_NEUTRAL_RATIO = 0.35

# این قفل فعلاً در تبدیل به ابر فعال نمی‌شود؛ فقط برای آینده نگه داشته شده.
RAIN_CHIN_RELEASE_STRENGTH_RATIO = 0.35
RAIN_CHIN_RELEASE_PROGRESS = 0.12

CHIN_FEATURE_WEIGHTS = np.ones(15, dtype=np.float32)

# -----------------------------
# Scapular Elevation settings
# -----------------------------
SHOULDER_REQUIRED_HOLD_TIME = 5.0

# اگر تشخیص شانه لحظه‌ای خراب شد، تایمر قطع نشود.
SHOULDER_RELEASE_CONFIRM_TIME = 0.25
SHOULDER_REAL_DOWN_RATIO = 0.10

SHOULDER_MATCH_THRESHOLD = 2.00
SHOULDER_MIN_TARGET_STRENGTH = 0.012
SHOULDER_MIN_SINGLE_LIFT = 0.004

SHOULDER_MIN_PROGRESS = 0.15
SHOULDER_MAX_PROGRESS = 999.00

# فیلترهای حرکت بدن عملاً آزاد شده‌اند، چون کاربر ممکن است جلو/عقب شود یا بدن کمی کج شود.
SHOULDER_MAX_HEAD_VERTICAL_CHANGE = 999.0
SHOULDER_MAX_ROLL_CHANGE = 999.0
SHOULDER_MAX_BODY_DISTANCE_CHANGE = 999.0

# برای تشخیص شانه، سر لازم نیست خیلی دقیق مثل Neutral باشد.
SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE = 40.0
SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE = 40.0
SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE = 45.0

SHOULDER_FEATURE_SMOOTHING = 0.65
SHOULDER_ANGLE_SMOOTHING = 0.75

SHOULDER_NEUTRAL_AVERAGE_FRAMES = 20
SHOULDER_TARGET_AVERAGE_FRAMES = 5
SHOULDER_MIN_VISIBILITY = 0.30

SHOULDER_FEATURE_WEIGHTS = np.array([
    1.15,
    1.15,
    1.00,
], dtype=np.float32)

# -----------------------------
# Scapular Retraction settings - Step 4
# -----------------------------
# این حرکت برای باران استفاده می‌شود:
# عقب بردن شانه/کتف + زیاد شدن فاصله دو کف دست.
# کالیبراسیون آن خودکار است و دکمه جدا ندارد.
RETRACTION_NEUTRAL_CAPTURE_SECONDS = 2.2
RETRACTION_TARGET_CAPTURE_SECONDS = 1.7
RETRACTION_RELEASE_REQUIRED_SECONDS = 0.8
RETRACTION_SUCCESS_HOLD_SECONDS = 3.0
RETRACTION_GAME_HOLD_SECONDS = 10.0
RETRACTION_MISS_TOLERANCE_SECONDS = 0.35

RETRACTION_MIN_HAND_SIZE_PIXELS = 35
RETRACTION_MIN_FACE_WIDTH_PIXELS = 55

# دست‌ها باید بیرون عرض شانه باشند.
RETRACTION_MIN_HAND_OUTSIDE_SHOULDER_MARGIN = 0.08
RETRACTION_MAX_HAND_SHOULDER_Y_DISTANCE = 1.35
RETRACTION_MIN_SHOULDER_VISIBILITY = 0.45

# ذخیره خودکار هدف: فاصله دست‌ها باید زیاد شود، هر دو دست بیرون بروند، و دست‌ها جلوتر نیایند.
RETRACTION_MIN_AUTO_TARGET_GAP_INCREASE = 0.28
RETRACTION_MIN_AUTO_TARGET_LEFT_OUTWARD = 0.075
RETRACTION_MIN_AUTO_TARGET_RIGHT_OUTWARD = 0.075
RETRACTION_MIN_AUTO_TARGET_TOTAL_STRENGTH = 0.16
RETRACTION_MIN_AUTO_TARGET_BACK_SIZE_DECREASE = 0.015
RETRACTION_MAX_AUTO_TARGET_HAND_SIZE_GROWTH = 0.025
RETRACTION_MAX_AUTO_TARGET_HAND_AREA_GROWTH = 0.060

RETRACTION_RELEASE_PROGRESS_MAX = 0.22

# تشخیص نهایی در بازی / تست کالیبراسیون
RETRACTION_MIN_TARGET_STRENGTH = 0.13
RETRACTION_ENOUGH_MOVEMENT_RATIO = 0.35
RETRACTION_MATCH_THRESHOLD = 1.05
RETRACTION_PROGRESS_MIN = 0.55
RETRACTION_PROGRESS_MAX = 1.75
RETRACTION_SIDE_ERROR_MAX = 0.95

RETRACTION_MIN_DETECT_GAP_INCREASE = 0.24
RETRACTION_MIN_DETECT_LEFT_OUTWARD = 0.060
RETRACTION_MIN_DETECT_RIGHT_OUTWARD = 0.060
RETRACTION_DETECT_GAP_TARGET_RATIO = 0.72
RETRACTION_DETECT_SIDE_TARGET_RATIO = 0.58
RETRACTION_DETECT_BACK_TARGET_RATIO = 0.45
RETRACTION_MIN_DETECT_BACK_SIZE_DECREASE = 0.010
RETRACTION_MAX_DETECT_HAND_SIZE_GROWTH = 0.020
RETRACTION_MAX_DETECT_HAND_AREA_GROWTH = 0.055

RETRACTION_FEATURE_WEIGHTS = np.array([
    2.80,  # gap between palms
    2.00,  # left palm outward from face center
    2.00,  # right palm outward from face center
    0.35,  # average palm width / face width
    0.75,  # average hand diagonal / face width
    0.55,  # average hand area / face area
    0.10,  # left hand vertical position
    0.10,  # right hand vertical position
], dtype=np.float32)

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

POSE_NOSE = mp_pose.PoseLandmark.NOSE.value
POSE_LEFT_EAR = mp_pose.PoseLandmark.LEFT_EAR.value
POSE_RIGHT_EAR = mp_pose.PoseLandmark.RIGHT_EAR.value
POSE_LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
POSE_RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

# -----------------------------
# 3D face model points
# -----------------------------
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1)
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# -----------------------------
# General helpers
# -----------------------------
def angle_diff(current, reference):
    return (current - reference + 180) % 360 - 180


def smooth_value(new_value, old_value, smoothing):
    if old_value is None:
        return new_value
    return smoothing * old_value + (1 - smoothing) * new_value


def smooth_angle(new_angle, old_angle, smoothing):
    if old_angle is None:
        return new_angle

    diff = angle_diff(new_angle, old_angle)
    unwrapped_angle = old_angle + diff

    return smoothing * old_angle + (1 - smoothing) * unwrapped_angle


def safe_dist(a, b):
    return float(np.linalg.norm(a - b))


# -----------------------------
# Head pose functions
# -----------------------------
def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = math.sqrt(
        rotation_matrix[0, 0] * rotation_matrix[0, 0] +
        rotation_matrix[1, 0] * rotation_matrix[1, 0]
    )

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    return pitch, yaw, roll


def get_head_pose(landmarks, frame_width, frame_height):
    image_points = []

    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        x = lm.x * frame_width
        y = lm.y * frame_height
        image_points.append((x, y))

    image_points = np.array(image_points, dtype=np.float64)

    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)

    return pitch, yaw, roll


# -----------------------------
# Chin Tuck helpers
# -----------------------------
def get_face_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def get_eye_roll(landmarks, w, h):
    left_eye = get_face_xy(landmarks, 33, w, h)
    right_eye = get_face_xy(landmarks, 263, w, h)

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]

    eye_roll = math.degrees(math.atan2(dy, dx))
    return eye_roll


def extract_chin_tuck_features(landmarks, w, h):
    nose = get_face_xy(landmarks, 1, w, h)
    chin = get_face_xy(landmarks, 152, w, h)
    forehead = get_face_xy(landmarks, 10, w, h)

    left_face = get_face_xy(landmarks, 234, w, h)
    right_face = get_face_xy(landmarks, 454, w, h)

    left_mouth = get_face_xy(landmarks, 61, w, h)
    right_mouth = get_face_xy(landmarks, 291, w, h)
    mouth_center = (left_mouth + right_mouth) / 2.0

    lower_lip = get_face_xy(landmarks, 14, w, h)

    left_eye = get_face_xy(landmarks, 33, w, h)
    right_eye = get_face_xy(landmarks, 263, w, h)
    eye_center = (left_eye + right_eye) / 2.0

    face_width = safe_dist(left_face, right_face)
    face_height = safe_dist(forehead, chin)

    if face_width < 1:
        face_width = 1.0

    if face_height < 1:
        face_height = 1.0

    face_center = (left_face + right_face) / 2.0

    chin_x_rel = (chin[0] - face_center[0]) / face_width
    chin_y_rel = (chin[1] - face_center[1]) / face_width

    nose_x_rel = (nose[0] - face_center[0]) / face_width
    nose_y_rel = (nose[1] - face_center[1]) / face_width

    mouth_x_rel = (mouth_center[0] - face_center[0]) / face_width
    mouth_y_rel = (mouth_center[1] - face_center[1]) / face_width

    lower_lip_y_rel = (lower_lip[1] - face_center[1]) / face_width

    nose_chin_dist = safe_dist(nose, chin) / face_width
    mouth_chin_dist = safe_dist(mouth_center, chin) / face_width
    lower_lip_chin_dist = safe_dist(lower_lip, chin) / face_width
    nose_mouth_dist = safe_dist(nose, mouth_center) / face_width
    eye_chin_dist = safe_dist(eye_center, chin) / face_width
    forehead_chin_dist = safe_dist(forehead, chin) / face_width

    lower_face_ratio = mouth_chin_dist / (face_height / face_width)
    chin_eye_vertical_ratio = abs(chin[1] - eye_center[1]) / face_width

    features = np.array([
        chin_x_rel,
        chin_y_rel,
        nose_x_rel,
        nose_y_rel,
        mouth_x_rel,
        mouth_y_rel,
        lower_lip_y_rel,
        nose_chin_dist,
        mouth_chin_dist,
        lower_lip_chin_dist,
        nose_mouth_dist,
        eye_chin_dist,
        forehead_chin_dist,
        lower_face_ratio,
        chin_eye_vertical_ratio
    ], dtype=np.float32)

    return features, face_width


def chin_tuck_score(current_features, neutral_features, target_features):
    target_change = target_features - neutral_features
    current_change = current_features - neutral_features

    target_strength = float(np.linalg.norm(target_change))
    current_strength = float(np.linalg.norm(current_change))

    if target_strength < 1e-8:
        return 999.0, target_strength, current_strength, 0.0, 999.0

    error = float(np.linalg.norm(current_change - target_change))
    score = error / target_strength

    progress = float(np.dot(current_change, target_change) / (target_strength ** 2))
    projected_vector = progress * target_change
    side_error = float(np.linalg.norm(current_change - projected_vector) / target_strength)

    return score, target_strength, current_strength, progress, side_error


def is_simple_chin_tuck(
    current_features,
    neutral_features,
    target_features,
    current_pitch=None,
    neutral_pitch_value=None,
    current_yaw=None,
    neutral_yaw_value=None,
    current_roll=None,
    neutral_roll_value=None
):
    score, target_strength, current_strength, progress, side_error = chin_tuck_score(
        current_features,
        neutral_features,
        target_features
    )

    if target_strength < CHIN_MIN_TARGET_STRENGTH:
        return False, score, target_strength, current_strength, progress, side_error

    head_is_stable = True

    if (
        current_pitch is not None and
        neutral_pitch_value is not None and
        current_yaw is not None and
        neutral_yaw_value is not None and
        current_roll is not None and
        neutral_roll_value is not None
    ):
        pitch_delta = abs(angle_diff(current_pitch, neutral_pitch_value))
        yaw_delta = abs(angle_diff(current_yaw, neutral_yaw_value))
        roll_delta = abs(angle_diff(current_roll, neutral_roll_value))

        head_is_stable = (
            pitch_delta <= CHIN_HEAD_PITCH_LIMIT and
            yaw_delta <= CHIN_HEAD_YAW_LIMIT and
            roll_delta <= CHIN_HEAD_ROLL_LIMIT
        )

    enough_movement = current_strength >= (target_strength * CHIN_ENOUGH_MOVEMENT_RATIO)

    direct_match = score <= CHIN_MATCH_THRESHOLD

    direction_match = (
        progress >= CHIN_PROGRESS_MIN and
        progress <= CHIN_PROGRESS_MAX and
        side_error <= CHIN_SIDE_ERROR_MAX
    )

    detected = head_is_stable and enough_movement and (direct_match or direction_match)

    return detected, score, target_strength, current_strength, progress, side_error

# -----------------------------
# Scapular Elevation helpers
# -----------------------------
def get_pose_xy_visibility(landmarks, idx, w, h):
    lm = landmarks[idx]
    point = np.array([lm.x * w, lm.y * h], dtype=np.float32)
    visibility = lm.visibility
    return point, visibility


def extract_shoulder_lift_features(landmarks, w, h):
    nose, nose_vis = get_pose_xy_visibility(landmarks, POSE_NOSE, w, h)

    left_shoulder, left_shoulder_vis = get_pose_xy_visibility(
        landmarks,
        POSE_LEFT_SHOULDER,
        w,
        h
    )

    right_shoulder, right_shoulder_vis = get_pose_xy_visibility(
        landmarks,
        POSE_RIGHT_SHOULDER,
        w,
        h
    )

    left_ear, left_ear_vis = get_pose_xy_visibility(landmarks, POSE_LEFT_EAR, w, h)
    right_ear, right_ear_vis = get_pose_xy_visibility(landmarks, POSE_RIGHT_EAR, w, h)

    if nose_vis < SHOULDER_MIN_VISIBILITY:
        return None, None

    if left_shoulder_vis < SHOULDER_MIN_VISIBILITY or right_shoulder_vis < SHOULDER_MIN_VISIBILITY:
        return None, None

    shoulder_width = safe_dist(left_shoulder, right_shoulder)

    if shoulder_width < 40:
        return None, None

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0

    left_ref = left_ear if left_ear_vis >= 0.30 else nose
    right_ref = right_ear if right_ear_vis >= 0.30 else nose

    # در تصویر y کمتر یعنی بالاتر.
    # وقتی شانه بالا می‌رود، فاصله شانه تا گوش/صورت کمتر می‌شود.
    left_gap = (left_shoulder[1] - left_ref[1]) / shoulder_width
    right_gap = (right_shoulder[1] - right_ref[1]) / shoulder_width
    middle_gap = (mid_shoulder[1] - nose[1]) / shoulder_width

    features = np.array([
        -left_gap,
        -right_gap,
        -middle_gap
    ], dtype=np.float32)

    shoulder_angle = math.degrees(
        math.atan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        )
    )

    meta = {
        "nose_y": float(nose[1]),
        "shoulder_width": float(shoulder_width),
        "shoulder_angle": float(shoulder_angle)
    }

    return features, meta


def shoulder_lift_metrics(current_features, neutral_features, target_features):
    target_change = (target_features - neutral_features) * SHOULDER_FEATURE_WEIGHTS
    current_change = (current_features - neutral_features) * SHOULDER_FEATURE_WEIGHTS

    target_strength = float(np.linalg.norm(target_change))
    current_strength = float(np.linalg.norm(current_change))

    if target_strength < 1e-8:
        return 0.0, 999.0, 999.0, target_strength, current_strength

    progress = float(np.dot(current_change, target_change) / (target_strength ** 2))

    projected_vector = progress * target_change
    side_error = float(np.linalg.norm(current_change - projected_vector) / target_strength)

    direct_error = float(np.linalg.norm(current_change - target_change) / target_strength)

    return progress, side_error, direct_error, target_strength, current_strength



# -----------------------------
# Scapular Retraction helpers
# -----------------------------
def get_retraction_face_reference(face_results, frame_w, frame_h):
    """
    فقط از صورت برای نرمال‌سازی استفاده می‌کنیم؛ hip لازم نیست.
    """
    if not face_results.multi_face_landmarks:
        return None, None

    landmarks = face_results.multi_face_landmarks[0].landmark

    left_face = np.array([landmarks[234].x * frame_w, landmarks[234].y * frame_h], dtype=np.float32)
    right_face = np.array([landmarks[454].x * frame_w, landmarks[454].y * frame_h], dtype=np.float32)
    nose = np.array([landmarks[1].x * frame_w, landmarks[1].y * frame_h], dtype=np.float32)

    face_width = safe_dist(left_face, right_face)

    if face_width < RETRACTION_MIN_FACE_WIDTH_PIXELS:
        return None, None

    return nose, face_width


def get_retraction_shoulder_reference(pose_results, frame_w, frame_h):
    """
    فقط شانه‌ها را از Pose می‌گیریم؛ hip لازم نیست.
    خروجی بر اساس چپ/راست روی صفحه مرتب می‌شود.
    """
    if not pose_results.pose_landmarks:
        return None

    landmarks = pose_results.pose_landmarks.landmark

    left_lm = landmarks[POSE_LEFT_SHOULDER]
    right_lm = landmarks[POSE_RIGHT_SHOULDER]

    if (
        left_lm.visibility < RETRACTION_MIN_SHOULDER_VISIBILITY or
        right_lm.visibility < RETRACTION_MIN_SHOULDER_VISIBILITY
    ):
        return None

    p1 = np.array([left_lm.x * frame_w, left_lm.y * frame_h], dtype=np.float32)
    p2 = np.array([right_lm.x * frame_w, right_lm.y * frame_h], dtype=np.float32)

    if p1[0] <= p2[0]:
        screen_left = p1
        screen_right = p2
    else:
        screen_left = p2
        screen_right = p1

    shoulder_width = safe_dist(screen_left, screen_right)

    if shoulder_width < 40:
        return None

    return {
        "left": screen_left,
        "right": screen_right,
        "width": float(shoulder_width),
        "mid_y": float((screen_left[1] + screen_right[1]) / 2.0),
    }


def retraction_hand_to_info(hand_landmarks, frame_w, frame_h):
    points = []
    for lm in hand_landmarks.landmark:
        points.append([lm.x * frame_w, lm.y * frame_h, lm.z])

    points = np.array(points, dtype=np.float32)

    palm_indices = [0, 5, 9, 13, 17]
    palm_points = points[palm_indices, :2]

    center = np.mean(palm_points, axis=0)

    index_mcp = points[5, :2]
    middle_mcp = points[9, :2]
    pinky_mcp = points[17, :2]
    wrist = points[0, :2]

    palm_width = safe_dist(index_mcp, pinky_mcp)
    palm_height = safe_dist(wrist, middle_mcp)

    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)

    bbox_w = float(max_xy[0] - min_xy[0])
    bbox_h = float(max_xy[1] - min_xy[1])
    bbox_diag = math.sqrt(bbox_w * bbox_w + bbox_h * bbox_h)
    bbox_area = bbox_w * bbox_h

    hand_size = max(palm_width, palm_height, bbox_w, bbox_h)

    return {
        "landmarks": hand_landmarks,
        "points": points,
        "center": center,
        "palm_width": float(palm_width),
        "palm_height": float(palm_height),
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_diag": float(bbox_diag),
        "bbox_area": float(bbox_area),
        "hand_size": float(hand_size),
    }


def get_two_retraction_screen_hands(hand_results, frame_w, frame_h):
    """
    دو دست را بر اساس جایگاه روی تصویر مرتب می‌کند: دست سمت چپ تصویر، دست سمت راست تصویر.
    """
    if not hand_results.multi_hand_landmarks:
        return None, None, []

    infos = []

    for hand_lms in hand_results.multi_hand_landmarks:
        info = retraction_hand_to_info(hand_lms, frame_w, frame_h)
        if info["hand_size"] >= RETRACTION_MIN_HAND_SIZE_PIXELS:
            infos.append(info)

    if len(infos) < 2:
        return None, None, infos

    infos = sorted(infos, key=lambda item: item["hand_size"], reverse=True)[:2]
    infos = sorted(infos, key=lambda item: item["center"][0])

    return infos[0], infos[1], infos


def retraction_hands_are_outside_shoulders(left_hand, right_hand, shoulder_ref, face_width):
    if left_hand is None or right_hand is None or shoulder_ref is None or face_width is None:
        return False, {
            "left_margin": -999.0,
            "right_margin": -999.0,
            "left_y_distance": 999.0,
            "right_y_distance": 999.0,
            "required_margin": RETRACTION_MIN_HAND_OUTSIDE_SHOULDER_MARGIN,
            "max_y_distance": RETRACTION_MAX_HAND_SHOULDER_Y_DISTANCE,
            "left_ok": False,
            "right_ok": False,
            "y_ok": False,
        }

    face_width = max(float(face_width), 1.0)

    required_margin = RETRACTION_MIN_HAND_OUTSIDE_SHOULDER_MARGIN * face_width
    max_y_distance = RETRACTION_MAX_HAND_SHOULDER_Y_DISTANCE * face_width

    left_hand_center = left_hand["center"]
    right_hand_center = right_hand["center"]

    left_margin = float(shoulder_ref["left"][0] - left_hand_center[0])
    right_margin = float(right_hand_center[0] - shoulder_ref["right"][0])

    left_y_distance = abs(float(left_hand_center[1] - shoulder_ref["mid_y"]))
    right_y_distance = abs(float(right_hand_center[1] - shoulder_ref["mid_y"]))

    left_ok = left_margin >= required_margin
    right_ok = right_margin >= required_margin
    y_ok = (
        left_y_distance <= max_y_distance and
        right_y_distance <= max_y_distance
    )

    ok = left_ok and right_ok and y_ok

    return ok, {
        "left_margin": left_margin / face_width,
        "right_margin": right_margin / face_width,
        "left_y_distance": left_y_distance / face_width,
        "right_y_distance": right_y_distance / face_width,
        "required_margin": required_margin / face_width,
        "max_y_distance": max_y_distance / face_width,
        "left_ok": left_ok,
        "right_ok": right_ok,
        "y_ok": y_ok,
    }


def extract_retraction_features(left_hand, right_hand, face_center, face_width):
    if left_hand is None or right_hand is None or face_center is None or face_width is None:
        return None

    face_width = max(face_width, 1.0)

    left_center = left_hand["center"]
    right_center = right_hand["center"]

    palm_gap = abs(right_center[0] - left_center[0]) / face_width
    left_outward = (face_center[0] - left_center[0]) / face_width
    right_outward = (right_center[0] - face_center[0]) / face_width

    avg_palm_width = (left_hand["palm_width"] + right_hand["palm_width"]) / (2.0 * face_width)
    avg_hand_diag = (left_hand["bbox_diag"] + right_hand["bbox_diag"]) / (2.0 * face_width)
    avg_hand_area = (left_hand["bbox_area"] + right_hand["bbox_area"]) / (2.0 * face_width * face_width)

    left_y = (left_center[1] - face_center[1]) / face_width
    right_y = (right_center[1] - face_center[1]) / face_width

    return np.array([
        palm_gap,
        left_outward,
        right_outward,
        avg_palm_width,
        avg_hand_diag,
        avg_hand_area,
        left_y,
        right_y,
    ], dtype=np.float32)


def average_retraction_vectors(history):
    if len(history) == 0:
        return None
    return np.mean(np.array(list(history)), axis=0).astype(np.float32)


def retraction_movement_strength_from_neutral(current_features, neutral_features):
    change = (current_features - neutral_features) * RETRACTION_FEATURE_WEIGHTS
    return float(np.linalg.norm(change))


def retraction_delta_info(current_features, neutral_features, target_features=None):
    gap_delta = float(current_features[0] - neutral_features[0])
    left_delta = float(current_features[1] - neutral_features[1])
    right_delta = float(current_features[2] - neutral_features[2])

    palm_width_decrease = float(neutral_features[3] - current_features[3])
    hand_diag_decrease = float(neutral_features[4] - current_features[4])
    hand_area_decrease = float(neutral_features[5] - current_features[5])

    info = {
        "gap_delta": gap_delta,
        "left_delta": left_delta,
        "right_delta": right_delta,
        "palm_width_decrease": palm_width_decrease,
        "hand_diag_decrease": hand_diag_decrease,
        "hand_area_decrease": hand_area_decrease,
    }

    if target_features is not None:
        info["target_gap_delta"] = float(target_features[0] - neutral_features[0])
        info["target_left_delta"] = float(target_features[1] - neutral_features[1])
        info["target_right_delta"] = float(target_features[2] - neutral_features[2])
        info["target_hand_diag_decrease"] = float(neutral_features[4] - target_features[4])
        info["target_hand_area_decrease"] = float(neutral_features[5] - target_features[5])

    return info


def retraction_target_candidate_is_good(current_features, neutral_features):
    if current_features is None or neutral_features is None:
        return False, 0.0, 0.0, 0.0, 0.0

    info = retraction_delta_info(current_features, neutral_features)
    gap_delta = info["gap_delta"]
    left_delta = info["left_delta"]
    right_delta = info["right_delta"]
    strength = retraction_movement_strength_from_neutral(current_features, neutral_features)

    both_hands_outward = (
        left_delta >= RETRACTION_MIN_AUTO_TARGET_LEFT_OUTWARD and
        right_delta >= RETRACTION_MIN_AUTO_TARGET_RIGHT_OUTWARD
    )

    hands_not_moving_forward = (
        info["hand_diag_decrease"] >= -RETRACTION_MAX_AUTO_TARGET_HAND_SIZE_GROWTH and
        info["hand_area_decrease"] >= -RETRACTION_MAX_AUTO_TARGET_HAND_AREA_GROWTH
    )

    backward_size_cue = (
        info["hand_diag_decrease"] >= RETRACTION_MIN_AUTO_TARGET_BACK_SIZE_DECREASE or
        info["hand_area_decrease"] >= RETRACTION_MIN_AUTO_TARGET_BACK_SIZE_DECREASE
    )

    good = (
        gap_delta >= RETRACTION_MIN_AUTO_TARGET_GAP_INCREASE and
        both_hands_outward and
        hands_not_moving_forward and
        backward_size_cue and
        strength >= RETRACTION_MIN_AUTO_TARGET_TOTAL_STRENGTH
    )

    return good, gap_delta, left_delta, right_delta, strength


def retraction_score(current_features, neutral_features, target_features):
    target_change = (target_features - neutral_features) * RETRACTION_FEATURE_WEIGHTS
    current_change = (current_features - neutral_features) * RETRACTION_FEATURE_WEIGHTS

    target_strength = float(np.linalg.norm(target_change))
    current_strength = float(np.linalg.norm(current_change))

    if target_strength < 1e-8:
        return 999.0, target_strength, current_strength, 0.0, 999.0

    error = float(np.linalg.norm(current_change - target_change))
    score = error / target_strength

    progress = float(np.dot(current_change, target_change) / (target_strength ** 2))
    projected = progress * target_change
    side_error = float(np.linalg.norm(current_change - projected) / target_strength)

    return score, target_strength, current_strength, progress, side_error


def strict_retraction_gate(current_features, neutral_features, target_features):
    info = retraction_delta_info(current_features, neutral_features, target_features)

    target_gap = max(info["target_gap_delta"], 0.0)
    target_left = max(info["target_left_delta"], 0.0)
    target_right = max(info["target_right_delta"], 0.0)
    target_back_diag = max(info["target_hand_diag_decrease"], 0.0)
    target_back_area = max(info["target_hand_area_decrease"], 0.0)

    required_gap = max(RETRACTION_MIN_DETECT_GAP_INCREASE, target_gap * RETRACTION_DETECT_GAP_TARGET_RATIO)
    required_left = max(RETRACTION_MIN_DETECT_LEFT_OUTWARD, target_left * RETRACTION_DETECT_SIDE_TARGET_RATIO)
    required_right = max(RETRACTION_MIN_DETECT_RIGHT_OUTWARD, target_right * RETRACTION_DETECT_SIDE_TARGET_RATIO)

    gap_ok = info["gap_delta"] >= required_gap
    left_ok = info["left_delta"] >= required_left
    right_ok = info["right_delta"] >= required_right

    not_forward = (
        info["hand_diag_decrease"] >= -RETRACTION_MAX_DETECT_HAND_SIZE_GROWTH and
        info["hand_area_decrease"] >= -RETRACTION_MAX_DETECT_HAND_AREA_GROWTH
    )

    if target_back_diag >= RETRACTION_MIN_DETECT_BACK_SIZE_DECREASE:
        back_diag_ok = info["hand_diag_decrease"] >= max(
            RETRACTION_MIN_DETECT_BACK_SIZE_DECREASE,
            target_back_diag * RETRACTION_DETECT_BACK_TARGET_RATIO
        )
    else:
        back_diag_ok = info["hand_diag_decrease"] >= -RETRACTION_MAX_DETECT_HAND_SIZE_GROWTH

    if target_back_area >= RETRACTION_MIN_DETECT_BACK_SIZE_DECREASE:
        back_area_ok = info["hand_area_decrease"] >= max(
            RETRACTION_MIN_DETECT_BACK_SIZE_DECREASE,
            target_back_area * RETRACTION_DETECT_BACK_TARGET_RATIO
        )
    else:
        back_area_ok = info["hand_area_decrease"] >= -RETRACTION_MAX_DETECT_HAND_AREA_GROWTH

    back_ok = not_forward and (back_diag_ok or back_area_ok)
    strict_ok = gap_ok and left_ok and right_ok and back_ok

    return strict_ok, info, required_gap, required_left, required_right


def is_retraction(current_features, neutral_features, target_features):
    score, target_strength, current_strength, progress, side_error = retraction_score(
        current_features,
        neutral_features,
        target_features
    )

    if target_strength < RETRACTION_MIN_TARGET_STRENGTH:
        return False, score, target_strength, current_strength, progress, side_error

    strict_ok, info, required_gap, required_left, required_right = strict_retraction_gate(
        current_features,
        neutral_features,
        target_features
    )

    enough_movement = current_strength >= (target_strength * RETRACTION_ENOUGH_MOVEMENT_RATIO)
    direct_match = score <= RETRACTION_MATCH_THRESHOLD
    direction_match = (
        progress >= RETRACTION_PROGRESS_MIN and
        progress <= RETRACTION_PROGRESS_MAX and
        side_error <= RETRACTION_SIDE_ERROR_MAX
    )

    detected = strict_ok and enough_movement and (direct_match or direction_match)

    return detected, score, target_strength, current_strength, progress, side_error


def update_retraction_hold(is_detected, hold_start, last_seen):
    now = time.time()

    if is_detected:
        if hold_start is None:
            hold_start = now
        last_seen = now
        return hold_start, last_seen, now - hold_start, True, False

    if hold_start is not None and last_seen is not None:
        if now - last_seen <= RETRACTION_MISS_TOLERANCE_SECONDS:
            return hold_start, last_seen, now - hold_start, True, True

    return None, None, 0.0, False, False


def reset_retraction_calibration_state(clear_saved=True):
    global retraction_calibration_state
    global retraction_neutral_features
    global retraction_target_features
    global retraction_neutral_start
    global retraction_target_start
    global retraction_release_start
    global retraction_test_hold_start
    global retraction_test_last_seen
    global retraction_calibration_message
    global retraction_calibration_success

    retraction_calibration_state = "capture_neutral"
    retraction_neutral_start = None
    retraction_target_start = None
    retraction_release_start = None
    retraction_test_hold_start = None
    retraction_test_last_seen = None

    retraction_neutral_buffer.clear()
    retraction_target_buffer.clear()
    retraction_current_buffer.clear()

    retraction_calibration_message = "Show face, shoulders, and both palms. Keep palms outside shoulder width."
    retraction_calibration_success = False

    if clear_saved:
        retraction_neutral_features = None
        retraction_target_features = None


def update_auto_retraction_calibration(
    current_features,
    face_detected,
    palms_detected,
    shoulders_detected,
    hands_outside_shoulders,
    shoulder_gate_info
):
    """
    کالیبراسیون خودکار حرکت عقب بردن شانه/کتف.
    دکمه‌ای ندارد: حالت عادی، هدف، ریلکس و تست را خودش مرحله‌به‌مرحله می‌گیرد.
    """
    global retraction_calibration_state
    global retraction_neutral_features
    global retraction_target_features
    global retraction_neutral_start
    global retraction_target_start
    global retraction_release_start
    global retraction_test_hold_start
    global retraction_test_last_seen
    global retraction_calibration_message
    global retraction_calibration_success

    now = time.time()

    if retraction_calibration_success:
        retraction_calibration_message = "Retraction calibration complete. Press ENTER to start game."
        return retraction_calibration_message

    state = retraction_calibration_state

    if state == "capture_neutral":
        if current_features is not None and face_detected and palms_detected and shoulders_detected and hands_outside_shoulders:
            retraction_neutral_buffer.append(current_features.copy())

            if retraction_neutral_start is None:
                retraction_neutral_start = now

            elapsed = now - retraction_neutral_start
            retraction_calibration_message = (
                f"Palms detected. Hold neutral palms up: {elapsed:.1f}s / "
                f"{RETRACTION_NEUTRAL_CAPTURE_SECONDS:.1f}s"
            )

            if elapsed >= RETRACTION_NEUTRAL_CAPTURE_SECONDS:
                neutral = average_retraction_vectors(retraction_neutral_buffer)
                if neutral is not None:
                    retraction_neutral_features = neutral.copy()
                    retraction_calibration_state = "capture_target"
                    retraction_target_start = None
                    retraction_target_buffer.clear()
                    retraction_calibration_message = (
                        "Neutral palms saved. Retract shoulders/scapulae and move palms outward/back."
                    )
        else:
            retraction_neutral_start = None
            retraction_neutral_buffer.clear()
            if not face_detected:
                retraction_calibration_message = "Face not detected. Show your face."
            elif not palms_detected:
                retraction_calibration_message = "Show BOTH palms clearly."
            elif not shoulders_detected:
                retraction_calibration_message = "Shoulders not detected. Sit a little farther from camera."
            elif not hands_outside_shoulders:
                retraction_calibration_message = (
                    f"Put palms outside shoulder width | "
                    f"L {shoulder_gate_info['left_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f} | "
                    f"R {shoulder_gate_info['right_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f}"
                )
            else:
                retraction_calibration_message = "Show face, shoulders, and both palms clearly."

    elif state == "capture_target":
        neutral = retraction_neutral_features

        if current_features is not None and neutral is not None and shoulders_detected and hands_outside_shoulders:
            averaged_current = average_retraction_vectors(retraction_current_buffer)
            if averaged_current is None:
                averaged_current = current_features.copy()

            good, gap_delta, left_delta, right_delta, strength = retraction_target_candidate_is_good(
                averaged_current,
                neutral
            )

            if good:
                retraction_target_buffer.append(averaged_current.copy())

                if retraction_target_start is None:
                    retraction_target_start = now

                elapsed = now - retraction_target_start
                retraction_calibration_message = (
                    f"Good retraction. Keep holding: {elapsed:.1f}s / "
                    f"{RETRACTION_TARGET_CAPTURE_SECONDS:.1f}s"
                )

                if elapsed >= RETRACTION_TARGET_CAPTURE_SECONDS:
                    target = average_retraction_vectors(retraction_target_buffer)
                    if target is not None:
                        target_strength = retraction_movement_strength_from_neutral(target, neutral)
                        if target_strength >= RETRACTION_MIN_TARGET_STRENGTH:
                            retraction_target_features = target.copy()
                            retraction_calibration_state = "wait_release"
                            retraction_release_start = None
                            retraction_target_buffer.clear()
                            retraction_calibration_message = "Retraction target saved. Relax hands/shoulders once."
                        else:
                            retraction_target_start = None
                            retraction_target_buffer.clear()
                            retraction_calibration_message = "Movement too small. Increase palm distance and move shoulders back."
            else:
                retraction_target_start = None
                retraction_target_buffer.clear()
                retraction_calibration_message = (
                    f"Retract more | gap +{gap_delta:.2f}/{RETRACTION_MIN_AUTO_TARGET_GAP_INCREASE:.2f} | "
                    f"L +{left_delta:.2f}/{RETRACTION_MIN_AUTO_TARGET_LEFT_OUTWARD:.2f} | "
                    f"R +{right_delta:.2f}/{RETRACTION_MIN_AUTO_TARGET_RIGHT_OUTWARD:.2f}"
                )
        else:
            retraction_target_start = None
            retraction_target_buffer.clear()
            if current_features is None:
                retraction_calibration_message = "Keep face and both palms visible."
            elif not shoulders_detected:
                retraction_calibration_message = "Shoulders not detected. Sit farther so both shoulders are visible."
            elif not hands_outside_shoulders:
                retraction_calibration_message = (
                    f"Hands must stay outside shoulders | "
                    f"L {shoulder_gate_info['left_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f} | "
                    f"R {shoulder_gate_info['right_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f}"
                )
            else:
                retraction_calibration_message = "Keep face, shoulders and both palms visible."

    elif state == "wait_release":
        neutral = retraction_neutral_features
        target = retraction_target_features

        if current_features is not None and neutral is not None and target is not None:
            averaged_current = average_retraction_vectors(retraction_current_buffer)
            if averaged_current is None:
                averaged_current = current_features.copy()

            score, target_strength, current_strength, progress, side_error = retraction_score(
                averaged_current,
                neutral,
                target
            )

            if progress <= RETRACTION_RELEASE_PROGRESS_MAX:
                if retraction_release_start is None:
                    retraction_release_start = now

                elapsed = now - retraction_release_start
                retraction_calibration_message = (
                    f"Relaxing detected. Hold relaxed position: {elapsed:.1f}s / "
                    f"{RETRACTION_RELEASE_REQUIRED_SECONDS:.1f}s"
                )

                if elapsed >= RETRACTION_RELEASE_REQUIRED_SECONDS:
                    retraction_calibration_state = "test"
                    retraction_test_hold_start = None
                    retraction_test_last_seen = None
                    retraction_calibration_message = "Now retract shoulders and hold 3 seconds for test."
            else:
                retraction_release_start = None
                retraction_calibration_message = "Relax once before testing. Bring hands closer to neutral."
        else:
            retraction_release_start = None
            retraction_calibration_message = "Keep face and both palms visible."

    elif state == "test":
        neutral = retraction_neutral_features
        target = retraction_target_features

        if current_features is not None and neutral is not None and target is not None and shoulders_detected and hands_outside_shoulders:
            averaged_current = average_retraction_vectors(retraction_current_buffer)
            if averaged_current is None:
                averaged_current = current_features.copy()

            detected, score, target_strength, current_strength, progress, side_error = is_retraction(
                averaged_current,
                neutral,
                target
            )

            retraction_test_hold_start, retraction_test_last_seen, hold_time, hold_active, noise_ignored = update_retraction_hold(
                detected,
                retraction_test_hold_start,
                retraction_test_last_seen
            )

            if hold_active:
                if noise_ignored:
                    retraction_calibration_message = (
                        f"Retraction test: {hold_time:.1f}/{RETRACTION_SUCCESS_HOLD_SECONDS:.1f}s | noise ignored"
                    )
                else:
                    retraction_calibration_message = (
                        f"Retraction test: {hold_time:.1f}/{RETRACTION_SUCCESS_HOLD_SECONDS:.1f}s"
                    )

                if hold_time >= RETRACTION_SUCCESS_HOLD_SECONDS:
                    retraction_calibration_success = True
                    retraction_calibration_message = "Retraction calibration complete. Press ENTER to start game."
            else:
                strict_ok, gate_info, req_gap, req_left, req_right = strict_retraction_gate(
                    averaged_current,
                    neutral,
                    target
                )
                retraction_calibration_message = (
                    f"Not retraction | gap {gate_info['gap_delta']:.2f}/{req_gap:.2f} | "
                    f"L {gate_info['left_delta']:.2f}/{req_left:.2f} | "
                    f"R {gate_info['right_delta']:.2f}/{req_right:.2f}"
                )
        else:
            retraction_test_hold_start = None
            retraction_test_last_seen = None
            if current_features is None:
                retraction_calibration_message = "Keep face and both palms visible."
            elif not shoulders_detected:
                retraction_calibration_message = "Shoulders not detected. Sit farther so both shoulders are visible."
            elif not hands_outside_shoulders:
                retraction_calibration_message = (
                    f"Hands outside shoulders required | "
                    f"L {shoulder_gate_info['left_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f} | "
                    f"R {shoulder_gate_info['right_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f}"
                )
            else:
                retraction_calibration_message = "Keep face, shoulders and both palms visible."

    return retraction_calibration_message


def reset_locked_retraction_progress():
    global locked_retraction_hold_start
    global locked_retraction_last_seen_time
    global locked_retraction_total_time
    global locked_retraction_last_update_time

    locked_retraction_hold_start = None
    locked_retraction_last_seen_time = None
    locked_retraction_total_time = 0.0
    locked_retraction_last_update_time = None


def pause_locked_retraction_progress():
    global locked_retraction_hold_start
    global locked_retraction_last_seen_time
    global locked_retraction_last_update_time

    locked_retraction_hold_start = None
    locked_retraction_last_seen_time = None
    locked_retraction_last_update_time = None




def reset_locked_rain_sequence():
    """
    ریست کامل توالی باران قفل‌شده:
    - باران در حال پخش
    - مکث بعد از Stage 3
    """
    global locked_rain_sequence_active
    global locked_rain_flower_key
    global locked_rain_start_time
    global locked_stage3_pause_active
    global locked_stage3_pause_flower_key
    global locked_stage3_pause_start_time

    locked_rain_sequence_active = False
    locked_rain_flower_key = None
    locked_rain_start_time = 0.0

    locked_stage3_pause_active = False
    locked_stage3_pause_flower_key = None
    locked_stage3_pause_start_time = 0.0


def start_locked_rain_sequence(flower_key):
    """
    وقتی حرکت عقب بردن شانه/کتف ۳ ثانیه کامل شد، فقط باران را شروع می‌کنیم.
    هنوز گل Stage 3 نمی‌شود و کاراکتر به مرکز برنمی‌گردد.
    """
    global locked_rain_sequence_active
    global locked_rain_flower_key
    global locked_rain_start_time
    global locked_stage3_pause_active
    global locked_stage3_pause_flower_key
    global locked_stage3_pause_start_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y

    now = time.time()

    locked_rain_sequence_active = True
    locked_rain_flower_key = flower_key
    locked_rain_start_time = now

    locked_stage3_pause_active = False
    locked_stage3_pause_flower_key = None
    locked_stage3_pause_start_time = 0.0

    # ابر همان بالای گل می‌ماند و باران از همان نقطه شروع می‌شود.
    rain_effect_start_time = now
    rain_effect_x = float(sun_target_x)
    rain_effect_y = float(sun_target_y)

def set_flower_stage3_and_score(flower_key):
    """
    وقتی باران کامل شود، فقط همان گل قفل‌شده از Stage 2 به Stage 3 می‌رود و +1 امتیاز می‌گیرد.
    """
    global top_flower_stage
    global bottom_flower_stage
    global right_orchid_stage
    global south_east_bluebloom_stage
    global left_tulip_stage
    global south_west_peony_stage
    global score

    if flower_key == "top" and top_flower_stage == 2:
        top_flower_stage = 3
        score += 1
        return True

    if flower_key == "bottom" and bottom_flower_stage == 2:
        bottom_flower_stage = 3
        score += 1
        return True

    if flower_key == "right_orchid" and right_orchid_stage == 2:
        right_orchid_stage = 3
        score += 1
        return True

    if flower_key == "south_east_bluebloom" and south_east_bluebloom_stage == 2:
        south_east_bluebloom_stage = 3
        score += 1
        return True

    if flower_key == "left_tulip" and left_tulip_stage == 2:
        left_tulip_stage = 3
        score += 1
        return True

    if flower_key == "south_west_peony" and south_west_peony_stage == 2:
        south_west_peony_stage = 3
        score += 1
        return True

    return False



def update_locked_rain_sequence():
    """
    این تابع هر فریم در حالت قفل صدا زده می‌شود.
    اگر توالی باران فعال باشد:
    - اول ۲ ثانیه ابر همان بالا می‌ماند و باران می‌بارد.
    - بعد گل Stage 3 می‌شود و امتیاز اضافه می‌شود.
    - سپس ۱.۵ ثانیه ابر همان بالا می‌ماند.
    - بعد کاراکتر خورشید می‌شود و به مرکز بازی برمی‌گردد.
    """
    global locked_rain_sequence_active
    global locked_rain_flower_key
    global locked_rain_start_time
    global locked_stage3_pause_active
    global locked_stage3_pause_flower_key
    global locked_stage3_pause_start_time
    global active_character
    global sun_shining_start_time
    global cloud_activation_time
    global game_state
    global game_finished
    global win_message

    now = time.time()

    if locked_rain_sequence_active:
        flower_key = locked_rain_flower_key
        flower_name = get_flower_name(flower_key)
        elapsed = now - locked_rain_start_time

        if elapsed < RAIN_EFFECT_DURATION:
            # هنوز باران در حال پخش است. گل هنوز Stage 3 نمی‌شود.
            return (
                f"Rain is falling on {flower_name}: {elapsed:.1f}s / "
                f"{RAIN_EFFECT_DURATION:.1f}s"
            )

        # ۲ ثانیه باران تمام شد؛ حالا گل Stage 3 می‌شود.
        grew = set_flower_stage3_and_score(flower_key)

        locked_rain_sequence_active = False
        locked_rain_flower_key = None
        locked_rain_start_time = 0.0

        locked_stage3_pause_active = True
        locked_stage3_pause_flower_key = flower_key
        locked_stage3_pause_start_time = now

        if grew:
            return (
                f"{flower_name} is now Stage 3! +1 score. "
                f"Wait {POST_STAGE3_RETURN_DELAY:.1f}s..."
            )

        return f"Rain finished. Wait {POST_STAGE3_RETURN_DELAY:.1f}s..."

    if locked_stage3_pause_active:
        flower_key = locked_stage3_pause_flower_key
        flower_name = get_flower_name(flower_key)
        elapsed = now - locked_stage3_pause_start_time

        if elapsed < POST_STAGE3_RETURN_DELAY:
            remaining = POST_STAGE3_RETURN_DELAY - elapsed
            return (
                f"{flower_name} is fully grown. Returning to center in "
                f"{remaining:.1f}s..."
            )

        # مکث تمام شد؛ حالا کاراکتر آزاد می‌شود و خورشید به مرکز برمی‌گردد.
        reset_locked_rain_sequence()
        unlock_character_from_flower()

        active_character = "sun"
        sun_shining_start_time = 0
        cloud_activation_time = 0.0
        reset_character_to_game_center()

        if all_flowers_fully_grown():
            game_finished = True
            mark_stage_completed(5)
            finalize_session_save("completed")
            game_state = "win"
            win_message = "Stage 5 complete. All flowers are fully grown."
            return "Stage 5 complete. All flowers are fully grown. You won!"

        return "Returning to center. Move to the next flower."

    return None


def process_locked_cloud_retraction_rain(
    current_features,
    face_detected,
    palms_detected,
    shoulders_detected,
    hands_outside_shoulders,
    shoulder_gate_info
):
    """
    Stage 5 cumulative Scapular Retraction.
    Releasing the movement pauses the saved progress; it does not reset it.
    """
    global locked_retraction_total_time
    global locked_retraction_last_update_time

    required_time = RETRACTION_GAME_HOLD_SECONDS

    if not (
        retraction_neutral_features is not None and
        retraction_target_features is not None and
        retraction_calibration_success
    ):
        pause_locked_retraction_progress()
        return (
            f"Rain movement is not calibrated. Progress saved: "
            f"{locked_retraction_total_time:.1f}s / {required_time:.1f}s"
        )

    if not face_detected:
        pause_locked_retraction_progress()
        return f"Show your face. Progress saved: {locked_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not palms_detected:
        pause_locked_retraction_progress()
        return f"Show BOTH palms. Progress saved: {locked_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not shoulders_detected:
        pause_locked_retraction_progress()
        return f"Shoulders not detected. Progress saved: {locked_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not hands_outside_shoulders:
        pause_locked_retraction_progress()
        return (
            f"Move palms outside shoulder width | "
            f"L {shoulder_gate_info['left_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f} | "
            f"R {shoulder_gate_info['right_margin']:.2f}/{shoulder_gate_info['required_margin']:.2f} | "
            f"saved {locked_retraction_total_time:.1f}s/{required_time:.1f}s"
        )
    if current_features is None:
        pause_locked_retraction_progress()
        return f"Keep face, shoulders, and palms visible. Progress saved: {locked_retraction_total_time:.1f}s / {required_time:.1f}s"

    averaged_current = average_retraction_vectors(retraction_current_buffer)
    if averaged_current is None:
        averaged_current = current_features.copy()

    detected, score_value, target_strength, current_strength, progress, side_error = is_retraction(
        averaged_current,
        retraction_neutral_features,
        retraction_target_features
    )

    locked_retraction_total_time, locked_retraction_last_update_time = update_cumulative_hold_progress(
        detected,
        locked_retraction_total_time,
        locked_retraction_last_update_time,
        required_time
    )

    locked_name = get_flower_name(locked_flower_key)

    if locked_retraction_total_time >= required_time:
        start_locked_rain_sequence(locked_flower_key)
        reset_locked_retraction_progress()
        return f"Rain started above {locked_name}! Cloud stays locked for {RAIN_EFFECT_DURATION:.1f}s."

    if detected:
        return f"Scapular Retraction total: {locked_retraction_total_time:.1f}s / {required_time:.1f}s"

    strict_ok, gate_info, req_gap, req_left, req_right = strict_retraction_gate(
        averaged_current,
        retraction_neutral_features,
        retraction_target_features
    )
    return (
        f"Retraction paused; progress saved {locked_retraction_total_time:.1f}s/{required_time:.1f}s | "
        f"gap {gate_info['gap_delta']:.2f}/{req_gap:.2f} | "
        f"L {gate_info['left_delta']:.2f}/{req_left:.2f} | "
        f"R {gate_info['right_delta']:.2f}/{req_right:.2f}"
    )



def draw_retraction_calibration_guides(
    frame,
    hand_results,
    shoulder_ref,
    face_center,
    face_width,
    left_hand,
    right_hand,
    hands_outside_shoulders
):
    """
    Clean calibration guide for palm/scapular retraction.
    To keep the calibration screen game-like and not crowded, we only draw:
    - center point of left palm
    - center point of right palm
    - distance line between the two palm centers

    The full hand skeleton is intentionally hidden here.
    """
    if left_hand is None or right_hand is None:
        return frame

    lc = left_hand["center"]
    rc = right_hand["center"]

    color = (70, 220, 110) if hands_outside_shoulders else (80, 90, 255)

    cv2.line(
        frame,
        (int(lc[0]), int(lc[1])),
        (int(rc[0]), int(rc[1])),
        color,
        4,
        cv2.LINE_AA
    )

    cv2.circle(frame, (int(lc[0]), int(lc[1])), 15, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(rc[0]), int(rc[1])), 15, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.circle(frame, (int(lc[0]), int(lc[1])), 10, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(rc[0]), int(rc[1])), 10, color, -1, cv2.LINE_AA)

    distance_px = safe_dist(lc, rc)
    mid_x = int((lc[0] + rc[0]) / 2)
    mid_y = int((lc[1] + rc[1]) / 2)

    label = f"Palm distance: {distance_px:.0f}px"
    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    label_x1 = int(mid_x - text_size[0] / 2 - 12)
    label_y1 = int(mid_y - 36)
    label_x2 = int(mid_x + text_size[0] / 2 + 12)
    label_y2 = int(mid_y - 8)

    frame_h, frame_w = frame.shape[:2]
    if label_x1 < 8:
        shift = 8 - label_x1
        label_x1 += shift
        label_x2 += shift
    if label_x2 > frame_w - 8:
        shift = label_x2 - (frame_w - 8)
        label_x1 -= shift
        label_x2 -= shift
    if label_y1 < 8:
        label_y1 = int(mid_y + 16)
        label_y2 = label_y1 + 28

    draw_filled_rounded_rect(frame, label_x1, label_y1, label_x2, label_y2, (35, 45, 55), radius=12)

    cv2.putText(
        frame,
        label,
        (label_x1 + 10, label_y2 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


def process_shoulder_toggle(current_pitch, current_yaw, current_roll, current_shoulder_features, current_shoulder_meta):
    """
    با ۵ ثانیه بالا گرفتن شانه‌ها، کاراکتر بین خورشید و ابر جابه‌جا می‌شود.

    نسخه اصلاح‌شده:
    - سر خیلی سخت‌گیرانه چک نمی‌شود.
    - اگر یک شانه کمی کمتر بالا بیاید، باز هم قبول می‌کند.
    - progress و side_error شرط اصلی نیستند.
    - اگر تشخیص شانه لحظه‌ای قطع شود، تایمر فقط وقتی ریست می‌شود
      که شانه‌ها واقعاً پایین آمده باشند.
    """
    global shoulder_hold_start
    global shoulder_release_start_time
    global shoulder_toggle_waiting_release
    global active_character
    global sun_shining_start_time
    global rain_effect_start_time
    global rain_chin_hold_start
    global rain_chin_last_seen_time
    global rain_waiting_for_chin_release
    global cloud_activation_time

    if active_character == "sun":
        shoulder_status = "Scapular Elevation 5s -> cloud"
    else:
        shoulder_status = "Scapular Elevation 5s -> sun"

    head_ready_for_shoulder_toggle = False
    shoulder_head_pitch_delta = 999.0
    shoulder_head_yaw_delta = 999.0
    shoulder_head_roll_delta = 999.0

    if (
        current_pitch is not None and
        current_yaw is not None and
        current_roll is not None and
        neutral_pitch is not None and
        neutral_yaw is not None and
        neutral_roll is not None
    ):
        shoulder_head_pitch_delta = abs(angle_diff(current_pitch, neutral_pitch))
        shoulder_head_yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))
        shoulder_head_roll_delta = abs(angle_diff(current_roll, neutral_roll))

        head_ready_for_shoulder_toggle = (
            shoulder_head_pitch_delta <= SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE and
            shoulder_head_yaw_delta <= SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE and
            shoulder_head_roll_delta <= SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE
        )

    if not head_ready_for_shoulder_toggle:
        shoulder_hold_start = None
        shoulder_release_start_time = None
        return (
            f"Keep head almost straight for shoulder switch | "
            f"pitch: {shoulder_head_pitch_delta:.1f}"
        )

    if not (
        current_shoulder_features is not None and
        current_shoulder_meta is not None and
        shoulder_neutral_features is not None and
        shoulder_target_features is not None and
        shoulder_neutral_width is not None and
        shoulder_neutral_nose_y is not None and
        shoulder_neutral_angle is not None
    ):
        shoulder_hold_start = None
        shoulder_release_start_time = None
        return shoulder_status

    sh_progress, sh_side_error, sh_direct_error, sh_target_strength, sh_current_strength = shoulder_lift_metrics(
        current_shoulder_features,
        shoulder_neutral_features,
        shoulder_target_features
    )

    head_vertical_change = abs(
        current_shoulder_meta["nose_y"] - shoulder_neutral_nose_y
    ) / max(shoulder_neutral_width, 1.0)

    body_distance_change = abs(
        current_shoulder_meta["shoulder_width"] - shoulder_neutral_width
    ) / max(shoulder_neutral_width, 1.0)

    shoulder_roll_change = abs(
        angle_diff(current_shoulder_meta["shoulder_angle"], shoulder_neutral_angle)
    )

    target_left_lift = shoulder_target_features[0] - shoulder_neutral_features[0]
    target_right_lift = shoulder_target_features[1] - shoulder_neutral_features[1]

    current_left_lift = current_shoulder_features[0] - shoulder_neutral_features[0]
    current_right_lift = current_shoulder_features[1] - shoulder_neutral_features[1]

    # برای جلوگیری از تشخیص اشتباهی:
    # شروع حرکت شانه باید سخت‌تر باشد.
    # اما بعد از شروع تایمر، ادامه حرکت کمی نرم‌تر گرفته می‌شود.

    SHOULDER_START_RATIO = 0.35
    SHOULDER_START_WEAK_RATIO = 0.22

    SHOULDER_CONTINUE_RATIO = 0.20
    SHOULDER_CONTINUE_WEAK_RATIO = 0.12

    SHOULDER_DOWN_RATIO = 0.20

    safe_target_left_lift = max(abs(target_left_lift), SHOULDER_MIN_SINGLE_LIFT)
    safe_target_right_lift = max(abs(target_right_lift), SHOULDER_MIN_SINGLE_LIFT)

    left_required_lift = max(
        SHOULDER_MIN_SINGLE_LIFT,
        safe_target_left_lift * SHOULDER_START_RATIO
    )

    right_required_lift = max(
        SHOULDER_MIN_SINGLE_LIFT,
        safe_target_right_lift * SHOULDER_START_RATIO
    )

    left_weak_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.75,
        safe_target_left_lift * SHOULDER_START_WEAK_RATIO
    )

    right_weak_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.75,
        safe_target_right_lift * SHOULDER_START_WEAK_RATIO
    )

    left_continue_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.75,
        safe_target_left_lift * SHOULDER_CONTINUE_RATIO
    )

    right_continue_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.75,
        safe_target_right_lift * SHOULDER_CONTINUE_RATIO
    )

    left_continue_weak_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.5,
        safe_target_left_lift * SHOULDER_CONTINUE_WEAK_RATIO
    )

    right_continue_weak_lift = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.5,
        safe_target_right_lift * SHOULDER_CONTINUE_WEAK_RATIO
    )

    left_ok = current_left_lift >= left_required_lift
    right_ok = current_right_lift >= right_required_lift

    both_shoulders_up = left_ok and right_ok

    one_strong_one_weak = (
        (current_left_lift >= left_required_lift and current_right_lift >= right_weak_lift) or
        (current_right_lift >= right_required_lift and current_left_lift >= left_weak_lift)
    )

    shoulders_start_accepted = both_shoulders_up or one_strong_one_weak

    left_continue_ok = current_left_lift >= left_continue_lift
    right_continue_ok = current_right_lift >= right_continue_lift

    both_shoulders_continue = left_continue_ok and right_continue_ok

    one_continue_strong_one_weak = (
        (current_left_lift >= left_continue_lift and current_right_lift >= right_continue_weak_lift) or
        (current_right_lift >= right_continue_lift and current_left_lift >= left_continue_weak_lift)
    )

    shoulders_continue_accepted = both_shoulders_continue or one_continue_strong_one_weak

    left_down_limit = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.5,
        safe_target_left_lift * SHOULDER_DOWN_RATIO
    )

    right_down_limit = max(
        SHOULDER_MIN_SINGLE_LIFT * 0.5,
        safe_target_right_lift * SHOULDER_DOWN_RATIO
    )

    shoulders_really_down = (
        current_left_lift <= left_down_limit and
        current_right_lift <= right_down_limit
    )

    wrong_shoulder_movement = (
        head_vertical_change > SHOULDER_MAX_HEAD_VERTICAL_CHANGE or
        shoulder_roll_change > SHOULDER_MAX_ROLL_CHANGE or
        body_distance_change > SHOULDER_MAX_BODY_DISTANCE_CHANGE
    )

     # اگر تایمر هنوز شروع نشده، برای شروع حرکت سخت‌تر می‌گیریم.
    # اگر تایمر شروع شده، فقط برای ادامه کمی نرم‌تر می‌گیریم.
    if shoulder_hold_start is None:
        shoulder_detected_now = shoulders_start_accepted
    else:
        shoulder_detected_now = shoulders_continue_accepted

    raw_shoulder_lift = (
        sh_target_strength >= SHOULDER_MIN_TARGET_STRENGTH and
        shoulder_detected_now and
        (current_left_lift > 0 or current_right_lift > 0) and
        not wrong_shoulder_movement
    )

    shoulder_noise_ignored = False

    if raw_shoulder_lift:
        is_shoulder_lift = True
        shoulder_release_start_time = None

    elif shoulder_hold_start is not None:
        # نکته مهم:
        # قبلاً اگر شانه واقعاً پایین نبود، تایمر تا ابد ادامه پیدا می‌کرد.
        # حالا اگر تشخیص قطع شد، فقط 0.25 ثانیه نویز را تحمل می‌کنیم.
        # اگر بعد از 0.25 ثانیه دوباره شانه بالا تشخیص داده نشد، تایمر ریست می‌شود.
        if shoulder_release_start_time is None:
            shoulder_release_start_time = time.time()

        if time.time() - shoulder_release_start_time < SHOULDER_RELEASE_CONFIRM_TIME:
            is_shoulder_lift = True
            shoulder_noise_ignored = True
        else:
            is_shoulder_lift = False

    else:
        is_shoulder_lift = False

    if wrong_shoulder_movement:
        shoulder_hold_start = None
        shoulder_release_start_time = None
        shoulder_toggle_waiting_release = False
        return f"Wrong Shoulder movement | head: {head_vertical_change:.2f} | roll: {shoulder_roll_change:.1f}"

    # بعد از Toggle، تا وقتی شانه پایین نیاید، Toggle بعدی انجام نشود.
    if shoulder_toggle_waiting_release:
        if shoulders_really_down:
            shoulder_toggle_waiting_release = False
            shoulder_release_start_time = None
            return "Shoulders released. You can switch again."

        return "Release shoulders before next switch"

    if is_shoulder_lift:
        if shoulder_hold_start is None:
            shoulder_hold_start = time.time()

        shoulder_hold_time = time.time() - shoulder_hold_start
        next_character = "cloud" if active_character == "sun" else "sun"

        if shoulder_noise_ignored:
            shoulder_status = (
                f"Scapular Elevation: {shoulder_hold_time:.1f}s / "
                f"{SHOULDER_REQUIRED_HOLD_TIME:.1f}s -> {next_character} | noise ignored"
            )
        else:
            shoulder_status = (
                f"Scapular Elevation: {shoulder_hold_time:.1f}s / "
                f"{SHOULDER_REQUIRED_HOLD_TIME:.1f}s -> {next_character}"
            )

        if shoulder_hold_time >= SHOULDER_REQUIRED_HOLD_TIME:
            if active_character == "sun":
                active_character = "cloud"
                sun_shining_start_time = 0
                rain_effect_start_time = 0

                rain_chin_hold_start = None
                rain_chin_last_seen_time = None
                rain_waiting_for_chin_release = False

                cloud_activation_time = time.time()

    # وقتی خورشید به ابر تبدیل می‌شود،
    # فریم‌های قبلی Chin Tuck پاک می‌شوند تا با مرحله‌های بعدی قاطی نشود.
                clear_chin_histories()

                shoulder_status = "Cloud activated! Chin Tuck is disabled in cloud mode."
            else:
                active_character = "sun"
                rain_effect_start_time = 0
                rain_chin_hold_start = None
                rain_waiting_for_chin_release = False
                cloud_activation_time = 0.0

                shoulder_status = "Sun activated!"

            shoulder_hold_start = None
            shoulder_release_start_time = None
            shoulder_toggle_waiting_release = True

        return shoulder_status

    shoulder_hold_start = None
    shoulder_release_start_time = None

    return (
        f"Lift shoulders | "
        f"L: {current_left_lift:.3f}/{left_required_lift:.3f} | "
        f"R: {current_right_lift:.3f}/{right_required_lift:.3f} | "
        f"progress: {sh_progress:.2f}"
    )


# -----------------------------
# History helpers
# -----------------------------
def average_recent_vectors(history, max_frames):
    if len(history) == 0:
        return None

    recent = list(history)[-max_frames:]
    return np.mean(np.array(recent), axis=0).astype(np.float32)


def average_recent_values(history, max_frames):
    if len(history) == 0:
        return None

    recent = list(history)[-max_frames:]
    return float(np.mean(np.array(recent)))


def update_chin_hold_with_tolerance(is_detected, hold_start, last_seen_time):
    """
    اگر Chin Tuck برای لحظه کوتاه به خاطر نویز گم شود، تایمر صفر نمی‌شود.
    فقط اگر بیشتر از CHIN_MISS_TOLERANCE_TIME تشخیص قطع بماند، تایمر ریست می‌شود.
    """
    now = time.time()

    if is_detected:
        if hold_start is None:
            hold_start = now

        last_seen_time = now
        hold_time = now - hold_start

        return hold_start, last_seen_time, hold_time, True, False

    if (
        hold_start is not None and
        last_seen_time is not None and
        now - last_seen_time <= CHIN_MISS_TOLERANCE_TIME
    ):
        hold_time = now - hold_start

        return hold_start, last_seen_time, hold_time, True, True

    return None, None, 0.0, False, False


def reset_locked_chin_tuck_progress():
    """
    ریست شمارش Chin Tuck تجمعی برای گلی که تازه قفل شده است.
    """
    global locked_chin_tuck_total_time
    global locked_chin_tuck_last_update_time

    locked_chin_tuck_total_time = 0.0
    locked_chin_tuck_last_update_time = None


def update_locked_chin_tuck_progress(is_detected):
    """
    شمارش Chin Tuck در حالت قفل را به صورت تجمعی جلو می‌برد.
    وقتی Chin Tuck تشخیص داده شود، زمان اضافه می‌شود.
    وقتی تشخیص قطع شود، زمان صفر نمی‌شود؛ فقط اضافه شدن زمان متوقف می‌شود.
    """
    global locked_chin_tuck_total_time
    global locked_chin_tuck_last_update_time

    now = time.time()

    if is_detected:
        if locked_chin_tuck_last_update_time is None:
            locked_chin_tuck_last_update_time = now
        else:
            elapsed = now - locked_chin_tuck_last_update_time

            # اگر به هر دلیل برنامه برای لحظه طولانی گیر کرد، یک پرش زمانی بزرگ حساب نشود.
            if 0.0 <= elapsed <= 1.0:
                locked_chin_tuck_total_time += elapsed

            locked_chin_tuck_last_update_time = now

    else:
        # قطع حرکت، پیشرفت را صفر نمی‌کند؛ فقط شمارش را موقتاً متوقف می‌کند.
        locked_chin_tuck_last_update_time = None

    if locked_chin_tuck_total_time > LOCKED_CHIN_REQUIRED_TOTAL_TIME:
        locked_chin_tuck_total_time = LOCKED_CHIN_REQUIRED_TOTAL_TIME

    return locked_chin_tuck_total_time




def update_cumulative_hold_progress(is_detected, total_time, last_update_time, required_time):
    """
    Adds only correctly detected movement time to a cumulative timer.
    Releasing the movement pauses progress instead of resetting it.
    """
    now = time.time()

    if is_detected:
        if last_update_time is None:
            last_update_time = now
        else:
            elapsed = now - last_update_time
            if 0.0 <= elapsed <= 1.0:
                total_time += elapsed
            last_update_time = now
    else:
        last_update_time = None

    total_time = max(0.0, min(float(total_time), float(required_time)))
    return total_time, last_update_time

def reset_locked_shoulder_lift_progress():
    """Reset cumulative Scapular Elevation for a new flower or completed step."""
    global locked_shoulder_hold_start
    global locked_shoulder_release_start_time
    global locked_shoulder_total_time
    global locked_shoulder_last_update_time

    locked_shoulder_hold_start = None
    locked_shoulder_release_start_time = None
    locked_shoulder_total_time = 0.0
    locked_shoulder_last_update_time = None


def pause_locked_shoulder_lift_progress():
    """Pause Scapular Elevation without losing accumulated time."""
    global locked_shoulder_hold_start
    global locked_shoulder_release_start_time
    global locked_shoulder_last_update_time

    locked_shoulder_hold_start = None
    locked_shoulder_release_start_time = None
    locked_shoulder_last_update_time = None



def process_locked_shoulder_to_cloud(current_pitch, current_yaw, current_roll, current_shoulder_features, current_shoulder_meta):
    """Stage 5 cumulative Scapular Elevation: saved time resumes after release."""
    global locked_shoulder_total_time
    global locked_shoulder_last_update_time
    global active_character
    global sun_shining_start_time
    global rain_effect_start_time
    global rain_chin_hold_start
    global rain_chin_last_seen_time
    global rain_waiting_for_chin_release
    global cloud_activation_time

    required_time = LOCKED_SHOULDER_REQUIRED_HOLD_TIME

    if active_character == "cloud":
        reset_locked_shoulder_lift_progress()
        return "Cloud is locked above this flower. Now move shoulders back for rain."

    head_ready = False
    pitch_delta = yaw_delta = roll_delta = 999.0
    if (
        current_pitch is not None and current_yaw is not None and current_roll is not None and
        neutral_pitch is not None and neutral_yaw is not None and neutral_roll is not None
    ):
        pitch_delta = abs(angle_diff(current_pitch, neutral_pitch))
        yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))
        roll_delta = abs(angle_diff(current_roll, neutral_roll))
        head_ready = (
            pitch_delta <= SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE and
            yaw_delta <= SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE and
            roll_delta <= SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE
        )

    if not head_ready:
        pause_locked_shoulder_lift_progress()
        return (
            f"Keep head straight. Scapular Elevation progress saved: "
            f"{locked_shoulder_total_time:.1f}s / {required_time:.1f}s"
        )

    if not (
        current_shoulder_features is not None and current_shoulder_meta is not None and
        shoulder_neutral_features is not None and shoulder_target_features is not None and
        shoulder_neutral_width is not None and shoulder_neutral_nose_y is not None and
        shoulder_neutral_angle is not None
    ):
        pause_locked_shoulder_lift_progress()
        return f"Shoulders are not ready/visible. Progress saved: {locked_shoulder_total_time:.1f}s / {required_time:.1f}s"

    sh_progress, sh_side_error, sh_direct_error, sh_target_strength, sh_current_strength = shoulder_lift_metrics(
        current_shoulder_features,
        shoulder_neutral_features,
        shoulder_target_features
    )

    head_vertical_change = abs(current_shoulder_meta["nose_y"] - shoulder_neutral_nose_y) / max(shoulder_neutral_width, 1.0)
    body_distance_change = abs(current_shoulder_meta["shoulder_width"] - shoulder_neutral_width) / max(shoulder_neutral_width, 1.0)
    shoulder_roll_change = abs(angle_diff(current_shoulder_meta["shoulder_angle"], shoulder_neutral_angle))

    target_left = shoulder_target_features[0] - shoulder_neutral_features[0]
    target_right = shoulder_target_features[1] - shoulder_neutral_features[1]
    current_left = current_shoulder_features[0] - shoulder_neutral_features[0]
    current_right = current_shoulder_features[1] - shoulder_neutral_features[1]

    continuing = locked_shoulder_total_time > 0.0 or locked_shoulder_last_update_time is not None
    start_ratio = 0.35 if not continuing else 0.20
    weak_ratio = 0.22 if not continuing else 0.12
    safe_left = max(abs(target_left), SHOULDER_MIN_SINGLE_LIFT)
    safe_right = max(abs(target_right), SHOULDER_MIN_SINGLE_LIFT)
    left_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_left * start_ratio)
    right_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_right * start_ratio)
    left_weak = max(SHOULDER_MIN_SINGLE_LIFT * (0.75 if not continuing else 0.5), safe_left * weak_ratio)
    right_weak = max(SHOULDER_MIN_SINGLE_LIFT * (0.75 if not continuing else 0.5), safe_right * weak_ratio)

    detected = sh_target_strength >= SHOULDER_MIN_TARGET_STRENGTH and (
        (current_left >= left_req and current_right >= right_req) or
        (current_left >= left_req and current_right >= right_weak) or
        (current_right >= right_req and current_left >= left_weak)
    )

    wrong_movement = (
        head_vertical_change > SHOULDER_MAX_HEAD_VERTICAL_CHANGE or
        shoulder_roll_change > SHOULDER_MAX_ROLL_CHANGE or
        body_distance_change > SHOULDER_MAX_BODY_DISTANCE_CHANGE
    )
    if wrong_movement:
        detected = False

    locked_shoulder_total_time, locked_shoulder_last_update_time = update_cumulative_hold_progress(
        detected,
        locked_shoulder_total_time,
        locked_shoulder_last_update_time,
        required_time
    )

    if locked_shoulder_total_time >= required_time:
        active_character = "cloud"
        sun_shining_start_time = 0
        rain_effect_start_time = 0
        rain_chin_hold_start = None
        rain_chin_last_seen_time = None
        rain_waiting_for_chin_release = False
        cloud_activation_time = time.time()
        clear_chin_histories()
        reset_locked_shoulder_lift_progress()
        return "Cloud activated! Character stays locked above this flower. Now move shoulders back for 10 seconds."

    if detected:
        return f"Scapular Elevation total: {locked_shoulder_total_time:.1f}s / {required_time:.1f}s"

    return (
        f"Scapular Elevation paused; progress saved {locked_shoulder_total_time:.1f}s/{required_time:.1f}s | "
        f"L {current_left:.3f}/{left_req:.3f} | R {current_right:.3f}/{right_req:.3f} | progress {sh_progress:.2f}"
    )



def clear_chin_histories():
    chin_feature_history.clear()
    chin_face_width_history.clear()
    chin_pitch_history.clear()
    chin_yaw_history.clear()
    chin_eye_roll_history.clear()


def clear_shoulder_histories():
    shoulder_feature_history.clear()
    shoulder_nose_y_history.clear()
    shoulder_width_history.clear()
    shoulder_angle_history.clear()   
    
# -----------------------------
# Road / Side Bend movement helpers
# -----------------------------
# مسیر خاکی را با چند ناحیه مستطیلی تعریف می‌کنیم.
# مختصات‌ها براساس مرکز خورشید/ابر هستند، نه گوشه تصویر.
# اگر بعداً دیدی یک قسمت خاکی کمی بیرون مانده، فقط همین مستطیل‌ها را کمی بزرگ‌تر کن.

DIRT_ROAD_RECTS = [
    # جاده افقی اصلی وسط
    (60, 285, 1220, 455),

    # جاده عمودی وسط، از گلدان بالا تا گلدان پایین وسط
    (555, 110, 730, 665),

    # شاخه عمودی چپ، از جاده وسط به سمت گلدان جنوب غربی
    (185, 285, 405, 665),

    # شاخه پایینی چپ
    (60, 505, 520, 680),

    # شاخه عمودی راست، از جاده وسط به سمت گلدان جنوب شرقی
    (835, 285, 1065, 665),

    # شاخه پایینی راست
    (760, 505, 1220, 680),
]

# ناحیه عمودی وسط برای اینکه گلدان بالا/پایین وسط اشتباهی فعال نشود.
CENTER_VERTICAL_ROAD_CENTER_X_MIN = 555
CENTER_VERTICAL_ROAD_CENTER_X_MAX = 730


def get_character_center(x, y):
    return int(x + SUN_SIZE / 2), int(y + SUN_SIZE / 2)


def point_is_inside_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def is_point_on_dirt_road(px, py):
    for rect in DIRT_ROAD_RECTS:
        if point_is_inside_rect(px, py, rect):
            return True
    return False


def can_move_character_to(target_x, target_y):
    """
    Stage 5 uses free movement: the sun/cloud may move anywhere inside the
    visible game window. The other stages keep their own separate road checks.
    """
    return (
        SUN_MIN_X <= target_x <= SUN_MAX_X and
        SUN_MIN_Y <= target_y <= SUN_MAX_Y
    )


def character_is_on_center_vertical_road():
    """
    گلدان بالا و پایین وسط فقط وقتی فعال شوند که خورشید واقعاً روی مسیر عمودی وسط باشد.
    """
    center_x, center_y = get_character_center(sun_target_x, sun_target_y)

    return (
        CENTER_VERTICAL_ROAD_CENTER_X_MIN <= center_x <= CENTER_VERTICAL_ROAD_CENTER_X_MAX
    )


def character_is_on_horizontal_road():
    """
    برای سازگاری با بخش‌های قبلی کد نگه داشته شده.
    از این به بعد معیار اصلی، بودن روی هر ناحیه خاکی است.
    """
    return can_move_character_to(sun_target_x, sun_target_y)


def character_is_on_vertical_road():
    """
    برای سازگاری با بخش‌های قبلی کد نگه داشته شده.
    از این به بعد معیار اصلی، بودن روی هر ناحیه خاکی است.
    """
    return can_move_character_to(sun_target_x, sun_target_y)


def get_flower_name(flower_key):
    names = {
        "top": "Top flower",
        "bottom": "Red rose",
        "right_orchid": "Orchid",
        "south_east_bluebloom": "Bluebloom",
        "left_tulip": "Tulip",
        "south_west_peony": "Peony",
    }
    return names.get(flower_key, "Flower")


def get_flower_stage_value(flower_key):
    if flower_key == "top":
        return top_flower_stage
    if flower_key == "bottom":
        return bottom_flower_stage
    if flower_key == "right_orchid":
        return right_orchid_stage
    if flower_key == "south_east_bluebloom":
        return south_east_bluebloom_stage
    if flower_key == "left_tulip":
        return left_tulip_stage
    if flower_key == "south_west_peony":
        return south_west_peony_stage
    return 0


def all_flowers_fully_grown():
    """
    مرحله نهایی بازی:
    وقتی همه گل‌ها به Stage 3 برسند، صفحه برد فعال می‌شود.
    """
    return (
        top_flower_stage == 3 and
        bottom_flower_stage == 3 and
        right_orchid_stage == 3 and
        south_east_bluebloom_stage == 3 and
        left_tulip_stage == 3 and
        south_west_peony_stage == 3
    )


def set_flower_stage2(flower_key):
    """
    بعد از کامل شدن ۱۰ ثانیه Chin Tuck تجمعی، گل همان گلدان به Stage 2 می‌رود.
    در منطق جدید، این مرحله امتیاز ندارد؛ امتیاز فقط در Stage 3 اضافه می‌شود.
    """
    global top_flower_stage
    global bottom_flower_stage
    global right_orchid_stage
    global south_east_bluebloom_stage
    global left_tulip_stage
    global south_west_peony_stage

    global top_flower_animating
    global bottom_flower_animating
    global right_orchid_animating
    global south_east_bluebloom_animating
    global left_tulip_animating
    global south_west_peony_animating

    global top_flower_start_time
    global bottom_flower_start_time
    global right_orchid_start_time
    global south_east_bluebloom_start_time
    global left_tulip_start_time
    global south_west_peony_start_time

    global sun_shining_start_time

    if flower_key == "top" and top_flower_stage == 1:
        top_flower_stage = 2
        top_flower_animating = False
        top_flower_start_time = 0

    elif flower_key == "bottom" and bottom_flower_stage == 1:
        bottom_flower_stage = 2
        bottom_flower_animating = False
        bottom_flower_start_time = 0

    elif flower_key == "right_orchid" and right_orchid_stage == 1:
        right_orchid_stage = 2
        right_orchid_animating = False
        right_orchid_start_time = 0

    elif flower_key == "south_east_bluebloom" and south_east_bluebloom_stage == 1:
        south_east_bluebloom_stage = 2
        south_east_bluebloom_animating = False
        south_east_bluebloom_start_time = 0

    elif flower_key == "left_tulip" and left_tulip_stage == 1:
        left_tulip_stage = 2
        left_tulip_animating = False
        left_tulip_start_time = 0

    elif flower_key == "south_west_peony" and south_west_peony_stage == 1:
        south_west_peony_stage = 2
        south_west_peony_animating = False
        south_west_peony_start_time = 0

    sun_shining_start_time = time.time()


def get_flower_pot_position(flower_key):
    if flower_key == "top":
        return TOP_POT_CENTER_X, TOP_POT_SOIL_Y
    if flower_key == "bottom":
        return BOTTOM_POT_CENTER_X, BOTTOM_POT_SOIL_Y
    if flower_key == "right_orchid":
        return RIGHT_ORCHID_POT_CENTER_X, RIGHT_ORCHID_POT_SOIL_Y
    if flower_key == "south_east_bluebloom":
        return SOUTH_EAST_BLUEBLOOM_POT_CENTER_X, SOUTH_EAST_BLUEBLOOM_POT_SOIL_Y
    if flower_key == "left_tulip":
        return LEFT_TULIP_POT_CENTER_X, LEFT_TULIP_POT_SOIL_Y
    if flower_key == "south_west_peony":
        return SOUTH_WEST_PEONY_POT_CENTER_X, SOUTH_WEST_PEONY_POT_SOIL_Y

    return ROAD_CENTER_X + SUN_SIZE / 2, ROAD_CENTER_Y + SUN_SIZE / 2


def get_character_lock_position(flower_key):
    """
    مختصات خروجی، گوشه بالا-چپ خورشید/ابر است.
    مرکز خورشید روی X گلدان قرار می‌گیرد و Y کمی بالاتر از خاک گلدان می‌ایستد.
    """
    pot_center_x, pot_soil_y = get_flower_pot_position(flower_key)

    lock_x = int(pot_center_x - SUN_SIZE / 2)
    lock_y = int(pot_soil_y - SUN_SIZE - LOCK_GAP_ABOVE_POT)

    lock_x = max(0, min(WIDTH - SUN_SIZE, lock_x))
    lock_y = max(0, min(HEIGHT - SUN_SIZE, lock_y))

    return float(lock_x), float(lock_y)


def clear_all_movement_holds():
    """
    وقتی کاراکتر بالای گلدان قفل است، هیچ تایمر حرکتی قبلی نباید ادامه پیدا کند.
    """
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    global shoulder_hold_start
    global shoulder_release_start_time
    global shoulder_toggle_waiting_release
    global stage_chin_hold_start
    global stage_chin_last_seen_time
    global rain_chin_hold_start
    global rain_chin_last_seen_time
    global rain_waiting_for_chin_release

    flexion_hold_start = None
    extension_hold_start = None
    left_side_bend_hold_start = None
    right_side_bend_hold_start = None

    shoulder_hold_start = None
    shoulder_release_start_time = None
    shoulder_toggle_waiting_release = False

    stage_chin_hold_start = None
    stage_chin_last_seen_time = None
    rain_chin_hold_start = None
    rain_chin_last_seen_time = None
    rain_waiting_for_chin_release = False


def lock_character_above_flower(flower_key):
    """
    مرحله اول منطق جدید:
    خورشید دقیقاً بالای همان گلدان قرار می‌گیرد و قفل می‌شود.
    """
    global character_locked_to_flower
    global locked_flower_key
    global active_flower
    global sun_target_x
    global sun_target_y
    global active_character
    global rain_effect_start_time
    global sun_shining_start_time
    global cloud_activation_time

    lock_x, lock_y = get_character_lock_position(flower_key)

    character_locked_to_flower = True
    locked_flower_key = flower_key
    active_flower = flower_key

    # مرحله اول فقط با خورشید تست می‌شود.
    active_character = "sun"
    rain_effect_start_time = 0
    sun_shining_start_time = 0
    cloud_activation_time = 0.0

    sun_target_x = lock_x
    sun_target_y = lock_y

    clear_all_movement_holds()
    reset_locked_chin_tuck_progress()
    reset_locked_shoulder_lift_progress()
    reset_locked_retraction_progress()
    reset_locked_rain_sequence()


def unlock_character_from_flower():
    """
    برای مراحل بعدی نگه داشته شده است.
    فعلاً فقط با R همه‌چیز ریست می‌شود.
    """
    global character_locked_to_flower
    global locked_flower_key
    global active_flower

    character_locked_to_flower = False
    locked_flower_key = None
    active_flower = None
    reset_locked_chin_tuck_progress()
    reset_locked_shoulder_lift_progress()
    reset_locked_retraction_progress()
    reset_locked_rain_sequence()


def activate_flower_stage1_and_lock(flower_key):
    """
    اگر گل هنوز فعال نشده باشد، Stage 1 می‌شود.
    سپس خورشید بالای همان گلدان قفل می‌شود.
    در منطق جدید، Stage 1 امتیاز ندارد؛ امتیاز فقط برای Stage 3 است.
    """
    global top_flower_stage
    global bottom_flower_stage
    global right_orchid_stage
    global south_east_bluebloom_stage
    global left_tulip_stage
    global south_west_peony_stage

    global top_flower_animating
    global bottom_flower_animating
    global right_orchid_animating
    global south_east_bluebloom_animating
    global left_tulip_animating
    global south_west_peony_animating

    global top_flower_start_time
    global bottom_flower_start_time
    global right_orchid_start_time
    global south_east_bluebloom_start_time
    global left_tulip_start_time
    global south_west_peony_start_time

    flower_name = get_flower_name(flower_key)
    stage_before = get_flower_stage_value(flower_key)

    if flower_key == "top" and top_flower_stage == 0:
        top_flower_stage = 1
        top_flower_animating = False
        top_flower_start_time = 0

    elif flower_key == "bottom" and bottom_flower_stage == 0:
        bottom_flower_stage = 1
        bottom_flower_animating = False
        bottom_flower_start_time = 0

    elif flower_key == "right_orchid" and right_orchid_stage == 0:
        right_orchid_stage = 1
        right_orchid_animating = False
        right_orchid_start_time = 0

    elif flower_key == "south_east_bluebloom" and south_east_bluebloom_stage == 0:
        south_east_bluebloom_stage = 1
        south_east_bluebloom_animating = False
        south_east_bluebloom_start_time = 0

    elif flower_key == "left_tulip" and left_tulip_stage == 0:
        left_tulip_stage = 1
        left_tulip_animating = False
        left_tulip_start_time = 0

    elif flower_key == "south_west_peony" and south_west_peony_stage == 0:
        south_west_peony_stage = 1
        south_west_peony_animating = False
        south_west_peony_start_time = 0

    lock_character_above_flower(flower_key)

    if stage_before == 0:
        return f"{flower_name} Stage 1! Character locked above this flower."

    return f"{flower_name} already Stage {stage_before}. Character locked above this flower."


def reset_character_to_game_center():
    """
    After reaching a side pot, send the character back to the starting center.
    Both current and target are reset so the character does not visually cross the grass.
    """
    global sun_current_x
    global sun_current_y
    global sun_target_x
    global sun_target_y

    sun_current_x = float(sun_x)
    sun_current_y = float(sun_y)

    sun_target_x = float(sun_x)
    sun_target_y = float(sun_y)


def activate_side_flower_stage1(flower_key, flower_name):
    # برای سازگاری با اسم قدیمی تابع نگه داشته شده است.
    # از این به بعد، همه گل‌ها با منطق یکسان Stage 1 و Lock می‌شوند.
    return activate_flower_stage1_and_lock(flower_key)


def check_side_pot_reached():
    """
    Stage 5 six-pot proximity check.

    The character can approach every pot from any direction. Reaching the closest
    pot activates the same Stage 1 -> Chin Tuck -> Stage 2 -> Scapular Elevation
    -> cloud -> Scapular Retraction -> rain -> Stage 3 sequence as before.
    """
    if active_character != "sun" or character_locked_to_flower:
        return ""

    center_x, center_y = get_character_center(sun_target_x, sun_target_y)

    candidates = [
        "top",
        "bottom",
        "left_tulip",
        "right_orchid",
        "south_west_peony",
        "south_east_bluebloom",
    ]

    nearest_key = None
    nearest_score = None

    for flower_key in candidates:
        pot_x, pot_soil_y = get_flower_pot_position(flower_key)
        trigger_center_y = pot_soil_y + STAGE5_POT_TRIGGER_CENTER_Y_OFFSET
        dx = abs(center_x - pot_x)
        dy = abs(center_y - trigger_center_y)

        if (
            dx <= STAGE5_POT_TRIGGER_HALF_WIDTH and
            dy <= STAGE5_POT_TRIGGER_HALF_HEIGHT
        ):
            score_value = (dx / max(STAGE5_POT_TRIGGER_HALF_WIDTH, 1)) ** 2 + (
                dy / max(STAGE5_POT_TRIGGER_HALF_HEIGHT, 1)
            ) ** 2

            if nearest_score is None or score_value < nearest_score:
                nearest_score = score_value
                nearest_key = flower_key

    if nearest_key is None:
        return ""

    if get_flower_stage_value(nearest_key) >= 3:
        return f"{get_flower_name(nearest_key)} is already complete. Move to another pot."

    return activate_flower_stage1_and_lock(nearest_key)

def process_side_bend_movement(current_side_bend_angle, current_pitch, current_yaw):
    """
    Left side bend  -> character moves left
    Right side bend -> character moves right

    این تابع خورشید یا ابر را فقط روی نواحی خاکی حرکت می‌دهد.
    """
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    global sun_target_x
    global last_sun_move_time

    if character_locked_to_flower:
        left_side_bend_hold_start = None
        right_side_bend_hold_start = None
        return "Character is locked above the flower."

    if not (
        current_side_bend_angle is not None and
        neutral_side_bend_angle is not None and
        left_side_bend_direction is not None and
        left_side_bend_threshold is not None and
        right_side_bend_direction is not None and
        right_side_bend_threshold is not None
    ):
        left_side_bend_hold_start = None
        right_side_bend_hold_start = None
        return ""

    side_bend_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)

    left_amount = left_side_bend_direction * side_bend_delta
    right_amount = right_side_bend_direction * side_bend_delta

    # Ignore very tiny movements.
    side_try_amount = max(left_amount, right_amount)
    min_active_amount = min(left_side_bend_threshold, right_side_bend_threshold) * 0.45

    if side_try_amount < min_active_amount:
        left_side_bend_hold_start = None
        right_side_bend_hold_start = None
        return ""

    left_detected = False
    right_detected = False

    if left_amount >= left_side_bend_threshold:
        if left_side_bend_hold_start is None:
            left_side_bend_hold_start = time.time()

        if time.time() - left_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
            left_detected = True
    else:
        left_side_bend_hold_start = None

    if right_amount >= right_side_bend_threshold:
        if right_side_bend_hold_start is None:
            right_side_bend_hold_start = time.time()

        if time.time() - right_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
            right_detected = True
    else:
        right_side_bend_hold_start = None

    if not left_detected and not right_detected:
        return ""

    if time.time() - last_sun_move_time < SUN_MOVE_COOLDOWN:
        return ""

    if left_detected:
        candidate_x = max(
            SUN_MIN_X,
            sun_target_x - SUN_MOVE_DISTANCE
        )

        if not can_move_character_to(candidate_x, sun_target_y):
            left_side_bend_hold_start = None
            right_side_bend_hold_start = None
            return "Cannot move outside the game area."

        sun_target_x = candidate_x

        last_sun_move_time = time.time()
        left_side_bend_hold_start = None
        right_side_bend_hold_start = None

        pot_message = check_side_pot_reached()

        if pot_message != "":
            return pot_message

        return "LEFT SIDE BEND - Character moves left"

    if right_detected:
        candidate_x = min(
            SUN_MAX_X,
            sun_target_x + SUN_MOVE_DISTANCE
        )

        if not can_move_character_to(candidate_x, sun_target_y):
            left_side_bend_hold_start = None
            right_side_bend_hold_start = None
            return "Cannot move outside the game area."

        sun_target_x = candidate_x

        last_sun_move_time = time.time()
        left_side_bend_hold_start = None
        right_side_bend_hold_start = None

        pot_message = check_side_pot_reached()

        if pot_message != "":
            return pot_message

        return "RIGHT SIDE BEND - Character moves right"

    return ""


# -----------------------------
# Image helpers
# -----------------------------
def remove_checker_background(frame_bgr):
    """
    حذف بک‌گراند شطرنجی، سفید، کرم یا طوسی روشن برای گل‌ها و خورشید.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    b, g, r = cv2.split(frame_bgr)

    bg_like = (
        (
            ((s < 75) & (v > 120)) |
            ((s < 55) & (v > 175))
        )
        &
        (np.abs(b.astype(np.int16) - g.astype(np.int16)) < 60)
        &
        (np.abs(g.astype(np.int16) - r.astype(np.int16)) < 60)
        &
        (np.abs(b.astype(np.int16) - r.astype(np.int16)) < 60)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(bg_like, connectivity=8)

    background_mask = np.zeros_like(bg_like, dtype=np.uint8)

    for label in range(1, num_labels):
        component = labels == label

        touches_border = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )

        if touches_border:
            background_mask[component] = 1

    alpha = np.where(background_mask == 1, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    frame_bgra = cv2.merge([b, g, r, alpha])
    return frame_bgra


def remove_checker_background_cloud(frame_bgr):
    """
    حذف دقیق‌تر بک‌گراند شطرنجی اطراف ابر.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(frame_bgr)

    bg_like = (
        (s < 35)
        & (v > 175)
        & (np.abs(b.astype(np.int16) - g.astype(np.int16)) < 22)
        & (np.abs(g.astype(np.int16) - r.astype(np.int16)) < 22)
        & (np.abs(b.astype(np.int16) - r.astype(np.int16)) < 22)
    ).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    bg_like = cv2.morphologyEx(bg_like, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels = cv2.connectedComponents(bg_like, connectivity=8)
    background_mask = np.zeros_like(bg_like, dtype=np.uint8)

    for label in range(1, num_labels):
        component = labels == label

        touches_border = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )

        if touches_border:
            background_mask[component] = 1

    background_mask = cv2.dilate(background_mask, kernel, iterations=1)

    alpha = np.where(background_mask == 1, 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)

    return cv2.merge([b, g, r, alpha])


def overlay_transparent(background_img, object_bgra, x, y):
    obj_h, obj_w = object_bgra.shape[:2]

    if x < 0 or y < 0:
        return background_img

    if x + obj_w > background_img.shape[1] or y + obj_h > background_img.shape[0]:
        return background_img

    obj_rgb = object_bgra[:, :, :3]
    alpha = object_bgra[:, :, 3] / 255.0

    roi = background_img[y:y + obj_h, x:x + obj_w]

    for c in range(3):
        roi[:, :, c] = alpha * obj_rgb[:, :, c] + (1 - alpha) * roi[:, :, c]

    background_img[y:y + obj_h, x:x + obj_w] = roi
    return background_img


def draw_sun_glow(frame, x, y, size):
    center_x = int(x + size / 2)
    center_y = int(y + size / 2)

    pulse = 0.5 + 0.5 * math.sin(time.time() * 8)

    radius1 = int(size * (0.75 + 0.10 * pulse))
    radius2 = int(size * (1.05 + 0.12 * pulse))

    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius2, (0, 220, 255), -1)
    frame[:] = cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)

    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius1, (0, 255, 255), -1)
    frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        r_start = int(size * 0.62)
        r_end = int(size * 0.94)

        x1 = int(center_x + math.cos(rad) * r_start)
        y1 = int(center_y + math.sin(rad) * r_start)
        x2 = int(center_x + math.cos(rad) * r_end)
        y2 = int(center_y + math.sin(rad) * r_end)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

    return frame


def draw_rain(frame, x, y, size):
    """
    رسم باران زیر ابر.
    """
    cloud_center_x = int(x + size / 2)
    cloud_bottom_y = int(y + size - 10)

    rain_area_width = int(size * 1.25)
    start_x = int(cloud_center_x - rain_area_width / 2)

    t = time.time()

    for i in range(10):
        drop_x = start_x + i * int(rain_area_width / 9)
        offset = int((t * 180 + i * 23) % 90)
        drop_y = cloud_bottom_y + offset

        if drop_y > HEIGHT - 40:
            drop_y = cloud_bottom_y + (offset % 45)

        cv2.line(
            frame,
            (drop_x, drop_y),
            (drop_x - 8, drop_y + 22),
            (255, 180, 80),
            3
        )

    return frame


def load_png_asset(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Could not load asset: {path}")
        exit()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = remove_checker_background(img)

    elif len(img.shape) == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)

        if np.min(a) < 250:
            img = cv2.merge([b, g, r, a])
        else:
            bgr = cv2.merge([b, g, r])
            img = remove_checker_background(bgr)

    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = remove_checker_background(img)

    if size is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def load_sun_frames():
    sun_sheet = cv2.imread(SUN_SHEET_PATH)

    if sun_sheet is None:
        print("Sun sheet not found.")
        print("Make sure sun_sheet.png is in the same folder.")
        exit()

    sheet_h, sheet_w, _ = sun_sheet.shape

    cols = 3
    rows = 3

    frame_w = sheet_w // cols
    frame_h = sheet_h // rows

    frames = []

    for row in range(rows):
        for col in range(cols):
            x1 = col * frame_w
            y1 = row * frame_h
            x2 = x1 + frame_w
            y2 = y1 + frame_h

            frame = sun_sheet[y1:y2, x1:x2]
            frame = cv2.resize(frame, (SUN_SIZE, SUN_SIZE), interpolation=cv2.INTER_AREA)
            frame = remove_checker_background(frame)
            frames.append(frame)

    return frames


def load_cloud_frames():
    cloud_sheet = cv2.imread(CLOUD_SHEET_PATH, cv2.IMREAD_UNCHANGED)

    if cloud_sheet is None:
        print("Cloud sheet not found.")
        print("Make sure kawaii_cloud_faces_grid_pattern.png is in the same folder.")
        exit()

    sheet_h, sheet_w = cloud_sheet.shape[:2]

    cols = 3
    rows = 3

    frame_w = sheet_w // cols
    frame_h = sheet_h // rows

    frames = []

    for row in range(rows):
        for col in range(cols):
            x1 = col * frame_w
            y1 = row * frame_h
            x2 = x1 + frame_w
            y2 = y1 + frame_h

            frame = cloud_sheet[y1:y2, x1:x2]

            if len(frame.shape) == 3 and frame.shape[2] == 4:
                b, g, r, a = cv2.split(frame)

                if np.min(a) < 250:
                    frame_bgra = cv2.merge([b, g, r, a])
                    frame_bgra = cv2.resize(
                        frame_bgra,
                        (SUN_SIZE, SUN_SIZE),
                        interpolation=cv2.INTER_AREA
                    )
                else:
                    bgr = cv2.merge([b, g, r])
                    bgr = cv2.resize(bgr, (SUN_SIZE, SUN_SIZE), interpolation=cv2.INTER_AREA)
                    frame_bgra = remove_checker_background_cloud(bgr)

            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.resize(frame, (SUN_SIZE, SUN_SIZE), interpolation=cv2.INTER_AREA)
                frame_bgra = remove_checker_background_cloud(frame_bgr)

            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                frame_bgr = cv2.resize(frame_bgr, (SUN_SIZE, SUN_SIZE), interpolation=cv2.INTER_AREA)
                frame_bgra = remove_checker_background_cloud(frame_bgr)

            frames.append(frame_bgra)

    return frames


def get_flower_asset(stage, flower1, flower2, flower3):
    if stage <= 0:
        return None
    elif stage == 1:
        return flower1
    elif stage == 2:
        return flower2
    else:
        return flower3


def update_flower_growth(stage, animating, start_time):
    if stage <= 0:
        return stage, False, 0.0

    return stage, False, 1.0


def draw_flower_on_pot(frame, flower_img, pot_center_x, pot_soil_y, growth_progress=1.0):
    growth_progress = max(0.0, min(1.0, growth_progress))

    scale = 0.35 + 0.65 * growth_progress

    draw_size = int(FLOWER_SIZE * scale)

    if draw_size < 5:
        return frame

    flower_resized = cv2.resize(
        flower_img,
        (draw_size, draw_size),
        interpolation=cv2.INTER_AREA
    )

    x = int(pot_center_x - draw_size / 2)
    y = int(pot_soil_y - draw_size + FLOWER_POT_OVERLAP)

    return overlay_transparent(frame, flower_resized, x, y)


def draw_text(frame, text, x, y, scale=0.8, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        3
    )


def draw_centered_text(frame, text, center_x, y, scale=0.8, color=(255, 255, 255), thickness=2):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_x = int(center_x - text_size[0] / 2)

    cv2.putText(
        frame,
        text,
        (text_x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_filled_rounded_rect(frame, x1, y1, x2, y2, color, radius=22):
    radius = max(0, min(radius, int((x2 - x1) / 2), int((y2 - y1) / 2)))

    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)

    cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
    cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
    cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
    cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)


def draw_rounded_rect(frame, x1, y1, x2, y2, color, radius=22, thickness=3):
    radius = max(0, min(radius, int((x2 - x1) / 2), int((y2 - y1) / 2)))

    cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_transparent_rounded_rect(frame, x1, y1, x2, y2, color, alpha=0.75, radius=26):
    overlay = frame.copy()
    draw_filled_rounded_rect(overlay, x1, y1, x2, y2, color, radius)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)



def load_optional_menu_image(path):
    """
    Loads an optional menu image. The game continues normally if the file is missing.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def load_first_optional_menu_image(paths):
    """
    Loads the first existing stage-selection background from a list of names.
    This lets the user save the design as backlevel.png / jpg / webp.
    """
    for path in paths:
        img = load_optional_menu_image(path)
        if img is not None:
            print(f"Loaded stage background: {path}")
            return img
    print("Optional stage background not found: backlevel")
    return None


def load_optional_png_asset(path, size=None):
    """
    Loads an optional transparent asset. If the file is missing, the game keeps running
    and uses a drawn fallback instead of crashing.
    """
    if not os.path.exists(path):
        print(f"Optional asset not found: {path}")
        return None

    try:
        return load_png_asset(path, size)
    except Exception as exc:
        print(f"Could not load optional asset {path}: {exc}")
        return None


def remove_white_background_for_tree(frame_bgra):
    """
    Removes the white/off-white background around the tutorial tree assets.
    This is intentionally used only for tree_stage1/2/3 so other game assets
    keep their previous behavior.
    """
    if frame_bgra is None:
        return None

    if len(frame_bgra.shape) == 2:
        bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_GRAY2BGR)
        original_alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    elif frame_bgra.shape[2] == 4:
        bgr = frame_bgra[:, :, :3].copy()
        original_alpha = frame_bgra[:, :, 3].copy()
    else:
        bgr = frame_bgra[:, :, :3].copy()
        original_alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    b_channel, g_channel, r_channel = cv2.split(bgr)

    # White / gray / warm-white candidates. Only border-connected regions are
    # removed, so pale flowers and highlights inside the tree stay safe.
    near_white = (
        ((s_channel < 60) & (v_channel > 178)) |
        (
            (b_channel > 188) & (g_channel > 188) & (r_channel > 188) &
            (np.abs(b_channel.astype(np.int16) - g_channel.astype(np.int16)) < 55) &
            (np.abs(g_channel.astype(np.int16) - r_channel.astype(np.int16)) < 55) &
            (np.abs(b_channel.astype(np.int16) - r_channel.astype(np.int16)) < 55)
        )
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    near_white = cv2.morphologyEx(near_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels = cv2.connectedComponents(near_white, connectivity=8)
    background_mask = np.zeros_like(near_white, dtype=np.uint8)

    for label in range(1, num_labels):
        component = labels == label
        touches_border = (
            component[0, :].any() or
            component[-1, :].any() or
            component[:, 0].any() or
            component[:, -1].any()
        )
        if touches_border:
            background_mask[component] = 1

    # Remove a thin white fringe around the tree.
    background_mask = cv2.dilate(background_mask, np.ones((3, 3), np.uint8), iterations=1)

    alpha = original_alpha.copy()
    alpha[background_mask == 1] = 0

    # Fully transparent pixels do not need white RGB values; this avoids white halos
    # after resizing and blending.
    bgr[background_mask == 1] = (0, 0, 0)

    return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], alpha])


def load_tree_png_asset(path, size=None):
    """
    Loads tutorial tree assets and cleans the white background around them.
    """
    if not os.path.exists(path):
        print(f"Optional tree asset not found: {path}")
        return None

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Could not load tree asset: {path}")
        return None

    img = remove_white_background_for_tree(img)

    if size is not None and img is not None:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    return img


def draw_menu_background(overlay_alpha=0.0):
    """
    Main menu background. If main_menu_background.png exists, it is used.
    Otherwise the normal game background is used.
    """
    if main_menu_background_img is not None:
        result = main_menu_background_img.copy()
    else:
        result = background.copy()

    if overlay_alpha > 0:
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (245, 250, 238), -1)
        cv2.addWeighted(overlay, overlay_alpha, result, 1 - overlay_alpha, 0, result)

    return result


def draw_fantasy_page_frame(title, subtitle=""):
    """
    Shared page frame for Profile, Progress, and Settings.
    It now uses the same fantasy garden background as the main menu, so all
    menu pages feel visually connected.
    """
    frame = draw_menu_background(overlay_alpha=0.10 if main_menu_background_img is not None else 0.25)

    # Soft readable layer in the middle. The main menu background stays visible,
    # but text and controls remain clean.
    draw_transparent_rounded_rect(frame, 100, 70, 1180, 650, (255, 248, 220), alpha=0.84, radius=38)
    draw_rounded_rect(frame, 100, 70, 1180, 650, (95, 145, 75), radius=38, thickness=5)

    # Title ribbon.
    draw_transparent_rounded_rect(frame, 360, 30, 920, 108, (40, 95, 135), alpha=0.90, radius=30)
    draw_rounded_rect(frame, 360, 30, 920, 108, (170, 220, 245), radius=30, thickness=3)
    draw_centered_text(frame, title, WIDTH // 2, 82, scale=1.04, color=(255, 250, 220), thickness=3)

    if subtitle != "":
        draw_centered_text(frame, subtitle, WIDTH // 2, 132, scale=0.48, color=(70, 90, 70), thickness=1)

    return frame


def draw_small_game_button(frame, button, hovered=False, selected=False, disabled=False, text_scale=0.52):
    """
    Smaller button used for Progress and Settings so labels do not leave their boxes.
    """
    x1, y1, x2, y2 = button["rect"]

    if disabled:
        fill = (202, 208, 205)
        border = (155, 165, 160)
        text_color = (105, 112, 110)
    elif selected:
        fill = (235, 165, 65)
        border = (255, 255, 255)
        text_color = (255, 255, 255)
    else:
        fill = (70, 170, 115)
        border = (255, 255, 255)
        text_color = (255, 255, 255)

    if hovered and not disabled:
        fill = (95, 210, 145) if not selected else (250, 185, 80)

    draw_transparent_rounded_rect(frame, x1 + 4, y1 + 5, x2 + 4, y2 + 5, (70, 80, 90), alpha=0.18, radius=20)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=20)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=20, thickness=3 if (hovered or selected) and not disabled else 2)

    label = button["label"]
    local_scale = text_scale
    while local_scale > 0.34:
        sz, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, local_scale, 2)
        if sz[0] <= (x2 - x1 - 22):
            break
        local_scale -= 0.03

    draw_centered_text(frame, label, int((x1 + x2) / 2), int((y1 + y2) / 2) + 8, scale=local_scale, color=text_color, thickness=2)


def draw_speaker_icon(frame, cx, cy, size=46, muted=False):
    """
    Draws a simple speaker icon using OpenCV shapes.
    """
    body_color = (255, 255, 255)
    wave_color = (255, 255, 255) if not muted else (90, 100, 110)
    s = int(size)

    pts = np.array([
        [cx - int(0.55 * s), cy - int(0.22 * s)],
        [cx - int(0.25 * s), cy - int(0.22 * s)],
        [cx + int(0.05 * s), cy - int(0.50 * s)],
        [cx + int(0.05 * s), cy + int(0.50 * s)],
        [cx - int(0.25 * s), cy + int(0.22 * s)],
        [cx - int(0.55 * s), cy + int(0.22 * s)],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], body_color, cv2.LINE_AA)

    if muted:
        cv2.line(frame, (cx + int(0.25 * s), cy - int(0.35 * s)), (cx + int(0.65 * s), cy + int(0.35 * s)), (80, 90, 230), 5, cv2.LINE_AA)
        cv2.line(frame, (cx + int(0.65 * s), cy - int(0.35 * s)), (cx + int(0.25 * s), cy + int(0.35 * s)), (80, 90, 230), 5, cv2.LINE_AA)
    else:
        cv2.ellipse(frame, (cx + int(0.08 * s), cy), (int(0.28 * s), int(0.28 * s)), 0, -45, 45, wave_color, 3, cv2.LINE_AA)
        cv2.ellipse(frame, (cx + int(0.08 * s), cy), (int(0.48 * s), int(0.48 * s)), 0, -45, 45, wave_color, 3, cv2.LINE_AA)
        cv2.ellipse(frame, (cx + int(0.08 * s), cy), (int(0.68 * s), int(0.68 * s)), 0, -45, 45, wave_color, 3, cv2.LINE_AA)


def draw_volume_bar(frame, x1, y1, x2, y2, value):
    value = max(0.0, min(1.0, float(value)))
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (230, 238, 225), radius=14)
    fill_x2 = int(x1 + (x2 - x1) * value)
    if fill_x2 > x1 + 6:
        draw_filled_rounded_rect(frame, x1, y1, fill_x2, y2, (70, 180, 115), radius=14)
    draw_rounded_rect(frame, x1, y1, x2, y2, (95, 145, 75), radius=14, thickness=2)



# -----------------------------
# Calibration UI helpers
# -----------------------------
def draw_ui_text(frame, text, x, y, scale=0.6, color=(40, 55, 65), thickness=2):
    cv2.putText(
        frame,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def wrap_text_lines(text, max_width, scale=0.62, thickness=2):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        candidate = word if current == "" else current + " " + word
        size, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

        if size[0] <= max_width or current == "":
            current = candidate
        else:
            lines.append(current)
            current = word

    if current != "":
        lines.append(current)

    return lines


def draw_wrapped_ui_text(frame, text, x, y, max_width, scale=0.62, color=(40, 55, 65), thickness=2, line_gap=28):
    lines = wrap_text_lines(text, max_width, scale, thickness)

    for i, line in enumerate(lines):
        draw_ui_text(
            frame,
            line,
            x,
            y + i * line_gap,
            scale=scale,
            color=color,
            thickness=thickness
        )

    return y + len(lines) * line_gap


def draw_status_pill(frame, x1, y1, x2, y2, text, fill_color, text_color=(255, 255, 255)):
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill_color, radius=16)

    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    tx = int((x1 + x2 - text_size[0]) / 2)
    ty = int(y1 + (y2 - y1 + text_size[1]) / 2) - 2

    draw_ui_text(
        frame,
        text,
        tx,
        ty,
        scale=0.50,
        color=text_color,
        thickness=2
    )


def get_calibration_step_text(face_detected, upper_body_detected):
    """
    Returns game-like title, clear instruction, and key hint for the current calibration step.
    The movement logic is not changed; this only improves the visual guidance.
    """
    if not face_detected:
        return (
            "Camera Check",
            "Show your face clearly in the camera. Sit in the center of the frame and look forward.",
            "No key yet"
        )

    if not upper_body_detected:
        return (
            "Shoulders Check",
            "Move a little farther from the camera so your head and both shoulders are visible.",
            "No key yet"
        )

    if neutral_pitch is None:
        return (
            "Step 1 - Neutral Posture",
            "Sit straight, keep your face forward, relax your shoulders, then press SPACE to save your neutral position.",
            "Press SPACE"
        )

    if flexion_direction is None:
        return (
            "Step 2 - Forward Flexion",
            "Slowly bend your head forward and downward. Keep your face mostly forward, then press F.",
            "Press F"
        )

    if extension_direction is None:
        return (
            "Step 3 - Backward Extension",
            "Gently move your head backward. Do not turn left or right. When the posture is clear, press B.",
            "Press B"
        )

    if left_side_bend_direction is None:
        return (
            "Step 4 - Left Side Bend",
            "Bend your head toward your LEFT shoulder. Keep your face forward and avoid rotating your neck, then press A.",
            "Press A"
        )

    if right_side_bend_direction is None:
        return (
            "Step 5 - Right Side Bend",
            "Bend your head toward your RIGHT shoulder. Keep your face forward and avoid rotating your neck, then press D.",
            "Press D"
        )

    if chin_target_features is None:
        return (
            "Step 6 - Chin Tuck",
            "Pull your chin straight backward like making a double chin. Hold it for about one second, then press T.",
            "Press T"
        )

    if shoulder_target_features is None:
        return (
            "Step 7 - Scapular Elevation",
            "Lift both shoulders upward clearly. Keep your head almost straight. Hold for about one second, then press U.",
            "Press U"
        )

    if not retraction_calibration_success:
        if retraction_calibration_state == "capture_neutral":
            title = "Step 8 - Scapular Retraction: Neutral"
            instruction = "Show both palms clearly. Keep your palms outside shoulder width and hold the relaxed position."
        elif retraction_calibration_state == "capture_target":
            title = "Step 8 - Scapular Retraction: Target"
            instruction = "Retract your shoulders/scapulae and move both palms outward/back. Keep both palm centers visible."
        elif retraction_calibration_state == "wait_release":
            title = "Step 8 - Scapular Retraction: Relax"
            instruction = "Relax your shoulders and hands once. Bring your hands closer to the neutral position."
        elif retraction_calibration_state == "test":
            title = "Step 8 - Scapular Retraction: Test"
            instruction = "Repeat the retraction and hold it for three seconds. Only the two palm centers and their distance are shown."
        else:
            title = "Step 8 - Scapular Retraction"
            instruction = "Follow the palm-center guide on the webcam preview."

        if retraction_calibration_message != "":
            instruction = instruction + "  |  " + retraction_calibration_message

        return (title, instruction, "Auto")

    return (
        "Calibration Complete",
        "Great job. All calibration steps are saved. Press ENTER to start the game.",
        "Press ENTER"
    )


def get_calibration_progress_items():
    return [
        ("Neutral", "SPACE", neutral_pitch is not None),
        ("Flexion", "F", flexion_threshold is not None),
        ("Extension", "B", extension_threshold is not None),
        ("Left Bend", "A", left_side_bend_threshold is not None),
        ("Right Bend", "D", right_side_bend_threshold is not None),
        ("Chin Tuck", "T", chin_target_features is not None),
        ("Scapular Elevation", "U", shoulder_target_features is not None),
        ("Scapular Retraction", "AUTO", retraction_calibration_success),
    ]


def get_current_calibration_index():
    items = get_calibration_progress_items()

    for i, item in enumerate(items):
        if not item[2]:
            return i

    return len(items)


def draw_calibration_progress_panel(
    frame,
    x1,
    y1,
    x2,
    y2,
    current_pitch,
    current_yaw,
    current_roll,
    current_side_bend_angle,
    face_detected,
    upper_body_detected
):
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (255, 252, 235), radius=28)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 170, 120), radius=28, thickness=3)

    draw_ui_text(frame, "Saved Positions", x1 + 26, y1 + 44, scale=0.72, color=(35, 115, 80), thickness=2)

    items = get_calibration_progress_items()
    current_index = get_current_calibration_index()

    row_y = y1 + 82
    row_h = 48

    for i, (label, key_name, saved) in enumerate(items):
        is_current = i == current_index

        if saved:
            fill = (218, 246, 224)
            border = (70, 180, 115)
            badge = "OK"
            badge_color = (60, 175, 105)
        elif is_current:
            fill = (255, 238, 190)
            border = (245, 175, 65)
            badge = "NOW"
            badge_color = (235, 155, 45)
        else:
            fill = (238, 241, 244)
            border = (190, 200, 205)
            badge = "--"
            badge_color = (165, 175, 182)

        draw_filled_rounded_rect(frame, x1 + 18, row_y, x2 - 18, row_y + row_h - 8, fill, radius=17)
        draw_rounded_rect(frame, x1 + 18, row_y, x2 - 18, row_y + row_h - 8, border, radius=17, thickness=2)

        draw_status_pill(frame, x1 + 28, row_y + 9, x1 + 76, row_y + 31, badge, badge_color)

        draw_ui_text(frame, label, x1 + 88, row_y + 27, scale=0.53, color=(45, 60, 70), thickness=2)
        draw_ui_text(frame, key_name, x2 - 82, row_y + 27, scale=0.48, color=(80, 95, 105), thickness=1)

        row_y += row_h

    live_y = y2 - 118
    draw_filled_rounded_rect(frame, x1 + 18, live_y, x2 - 18, y2 - 20, (245, 250, 255), radius=20)
    draw_rounded_rect(frame, x1 + 18, live_y, x2 - 18, y2 - 20, (145, 185, 210), radius=20, thickness=2)

    draw_ui_text(frame, "Live Check", x1 + 34, live_y + 30, scale=0.56, color=(50, 100, 140), thickness=2)

    face_text = "Face: OK" if face_detected else "Face: not detected"
    shoulder_text = "Shoulders: OK" if upper_body_detected else "Shoulders: not ready"

    face_color = (50, 160, 90) if face_detected else (60, 80, 230)
    shoulder_color = (50, 160, 90) if upper_body_detected else (60, 80, 230)

    draw_ui_text(frame, face_text, x1 + 34, live_y + 58, scale=0.47, color=face_color, thickness=1)
    draw_ui_text(frame, shoulder_text, x1 + 168, live_y + 58, scale=0.47, color=shoulder_color, thickness=1)

    if current_pitch is not None:
        draw_ui_text(
            frame,
            f"Pitch {current_pitch:.1f}   Yaw {current_yaw:.1f}   Roll {current_roll:.1f}",
            x1 + 34,
            live_y + 84,
            scale=0.43,
            color=(90, 100, 110),
            thickness=1
        )


def paste_image_fit(frame, image, x1, y1, x2, y2):
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    img_h, img_w = image.shape[:2]
    scale = min(box_w / img_w, box_h / img_h)

    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    px = x1 + int((box_w - new_w) / 2)
    py = y1 + int((box_h - new_h) / 2)

    frame[py:py + new_h, px:px + new_w] = resized

    return px, py, new_w, new_h


def draw_calibration_camera_panel(frame, camera_frame, x1, y1, x2, y2):
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), radius=30)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 170, 120), radius=30, thickness=3)

    draw_ui_text(frame, "Webcam Preview", x1 + 28, y1 + 43, scale=0.72, color=(35, 115, 80), thickness=2)

    inner_x1 = x1 + 24
    inner_y1 = y1 + 64
    inner_x2 = x2 - 24
    inner_y2 = y2 - 24

    draw_filled_rounded_rect(frame, inner_x1, inner_y1, inner_x2, inner_y2, (30, 35, 42), radius=24)

    # The camera is placed inside the right panel; it is resized to keep its aspect ratio.
    paste_image_fit(
        frame,
        camera_frame,
        inner_x1 + 10,
        inner_y1 + 10,
        inner_x2 - 10,
        inner_y2 - 10
    )



def get_calibration_buttons():
    return [
        {"id": "calibration_main_menu", "label": "Main Menu", "rect": (1018, 650, 1248, 705)},
    ]


def draw_calibration_bottom_buttons(frame):
    buttons = get_calibration_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)
    for button in buttons:
        draw_small_game_button(
            frame,
            button,
            hovered=(button["id"] == hover_button),
            selected=False,
            disabled=False,
            text_scale=0.54
        )

def draw_calibration_screen(
    camera_frame,
    face_detected,
    upper_body_detected,
    current_pitch,
    current_yaw,
    current_roll,
    current_side_bend_angle
):
    # Use the garden background with a soft light overlay so calibration feels like part of the game.
    frame = background.copy()
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (242, 250, 238), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)

    margin = 24
    gap = 20

    top_x1 = margin
    top_y1 = 20
    top_x2 = WIDTH - margin
    top_y2 = 145

    left_x1 = margin
    left_y1 = top_y2 + gap
    left_x2 = 365
    left_y2 = HEIGHT - margin

    cam_x1 = left_x2 + gap
    cam_y1 = left_y1
    cam_x2 = WIDTH - margin
    cam_y2 = HEIGHT - margin

    title, instruction, key_hint = get_calibration_step_text(face_detected, upper_body_detected)

    # Top instruction panel
    draw_filled_rounded_rect(frame, top_x1, top_y1, top_x2, top_y2, (255, 248, 220), radius=32)
    draw_rounded_rect(frame, top_x1, top_y1, top_x2, top_y2, (55, 170, 115), radius=32, thickness=4)

    draw_ui_text(
        frame,
        "Calibration",
        top_x1 + 30,
        top_y1 + 42,
        scale=0.86,
    color=(0, 135, 85),
    thickness=3
)

# عنوان مرحله را داخل ستون چپ محدود می‌کنیم تا وارد متن توضیح نشود
    draw_wrapped_ui_text(
      frame,
      title,
      top_x1 + 30,
      top_y1 + 75,
      max_width=285,
      scale=0.52,
      color=(45, 75, 65),
      thickness=2,
      line_gap=24
)

# متن توضیح را کمی بیشتر به راست می‌بریم
    instruction_x = top_x1 + 390

    draw_wrapped_ui_text(
      frame,
      instruction,
      instruction_x,
      top_y1 + 48,
      max_width=top_x2 - instruction_x - 245,
      scale=0.52,
      color=(45, 60, 65),
      thickness=2,
      line_gap=24
)

    done_count = sum(1 for item in get_calibration_progress_items() if item[2])
    total_count = len(get_calibration_progress_items())

    draw_status_pill(frame, top_x2 - 185, top_y1 + 29, top_x2 - 34, top_y1 + 65, key_hint, (65, 160, 220))
    draw_ui_text(frame, f"{done_count}/{total_count} saved", top_x2 - 171, top_y1 + 103, scale=0.55, color=(70, 95, 105), thickness=2)

    # Small progress bar
    bar_x1 = top_x2 - 190
    bar_y1 = top_y1 + 112
    bar_x2 = top_x2 - 34
    bar_y2 = top_y1 + 126
    draw_filled_rounded_rect(frame, bar_x1, bar_y1, bar_x2, bar_y2, (225, 232, 230), radius=7)

    if total_count > 0:
        fill_w = int((bar_x2 - bar_x1) * done_count / total_count)
        draw_filled_rounded_rect(frame, bar_x1, bar_y1, bar_x1 + fill_w, bar_y2, (70, 185, 115), radius=7)

    # Left saved-positions panel
    draw_calibration_progress_panel(
        frame,
        left_x1,
        left_y1,
        left_x2,
        left_y2,
        current_pitch,
        current_yaw,
        current_roll,
        current_side_bend_angle,
        face_detected,
        upper_body_detected
    )

    # Right webcam panel
    draw_calibration_camera_panel(frame, camera_frame, cam_x1, cam_y1, cam_x2, cam_y2)

    draw_centered_text(
        frame,
        "R: reset calibration   |   Q: quit",
        520,
        HEIGHT - 8,
        scale=0.45,
        color=(65, 75, 80),
        thickness=1
    )

    draw_calibration_bottom_buttons(frame)

    return frame


def get_win_buttons():
    button_w = 430
    button_h = 54
    center_x = WIDTH // 2
    x1 = int(center_x - button_w / 2)
    x2 = x1 + button_w

    y1 = 420
    gap = 68

    first_label = "Replay Stage" if current_stage_number == 5 else "Next Stage"

    return [
        {"id": "next_level", "label": first_label, "rect": (x1, y1, x2, y1 + button_h)},
        {"id": "main_menu", "label": "Back to Main Menu", "rect": (x1, y1 + gap, x2, y1 + gap + button_h)},
        {"id": "quit", "label": "Quit", "rect": (x1, y1 + 2 * gap, x2, y1 + 2 * gap + button_h)},
    ]


def point_inside_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def get_button_at_position(px, py, buttons):
    for button in buttons:
        if point_inside_rect(px, py, button["rect"]):
            return button["id"]
    return None


def get_home_button_rect():
    return HOME_ICON_BUTTON_RECT


def draw_home_icon_button(frame, hovered=False):
    """
    Small game-like home icon shown on the main game screen.
    The icon is drawn with OpenCV shapes so it does not depend on emoji fonts.
    """
    x1, y1, x2, y2 = get_home_button_rect()

    shadow = (80, 90, 100)
    fill = (255, 248, 220) if not hovered else (255, 255, 235)
    border = (45, 170, 115) if not hovered else (0, 210, 170)
    roof = (40, 120, 85)
    body = (70, 165, 120)

    draw_transparent_rounded_rect(frame, x1 + 5, y1 + 6, x2 + 5, y2 + 6, shadow, alpha=0.22, radius=20)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=20)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=20, thickness=3 if not hovered else 4)

    cx = int((x1 + x2) / 2)
    roof_top = y1 + 17
    roof_left = x1 + 15
    roof_right = x2 - 15
    wall_top = y1 + 36
    wall_bottom = y2 - 16

    # Roof
    cv2.line(frame, (roof_left, wall_top), (cx, roof_top), roof, 5, cv2.LINE_AA)
    cv2.line(frame, (cx, roof_top), (roof_right, wall_top), roof, 5, cv2.LINE_AA)

    # House body
    cv2.rectangle(frame, (x1 + 21, wall_top), (x2 - 21, wall_bottom), body, -1)
    cv2.rectangle(frame, (x1 + 21, wall_top), (x2 - 21, wall_bottom), roof, 2)

    # Door
    cv2.rectangle(frame, (cx - 6, y2 - 31), (cx + 6, wall_bottom), (255, 248, 220), -1)

    return frame


def get_pause_menu_buttons():
    button_w = 430
    button_h = 56
    center_x = WIDTH // 2
    x1 = int(center_x - button_w / 2)
    x2 = x1 + button_w

    y1 = 300
    gap = 70

    return [
        {"id": "continue", "label": "Continue Game", "rect": (x1, y1, x2, y1 + button_h)},
        {"id": "recalibrate", "label": "Recalibrate", "rect": (x1, y1 + gap, x2, y1 + gap + button_h)},
        {"id": "main_menu", "label": "Main Menu", "rect": (x1, y1 + 2 * gap, x2, y1 + 2 * gap + button_h)},
        {"id": "quit", "label": "Quit Game", "rect": (x1, y1 + 3 * gap, x2, y1 + 3 * gap + button_h)},
    ]


def draw_pause_menu_button(frame, button, hovered=False):
    x1, y1, x2, y2 = button["rect"]

    color_map = {
        "continue": ((70, 170, 110), (95, 210, 145)),
        "recalibrate": ((60, 165, 225), (85, 195, 255)),
        "main_menu": ((210, 145, 70), (235, 175, 95)),
        "quit": ((85, 120, 235), (110, 145, 255)),
    }

    normal_fill, hover_fill = color_map.get(button["id"], ((80, 150, 180), (110, 185, 215)))
    fill = hover_fill if hovered else normal_fill

    draw_transparent_rounded_rect(frame, x1 + 6, y1 + 7, x2 + 6, y2 + 7, (70, 80, 90), alpha=0.20, radius=24)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), radius=24, thickness=4 if hovered else 2)

    if hovered:
        cv2.circle(frame, (x1 + 28, int((y1 + y2) / 2)), 8, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x2 - 28, int((y1 + y2) / 2)), 8, (255, 255, 255), -1, cv2.LINE_AA)

    draw_centered_text(
        frame,
        button["label"],
        int((x1 + x2) / 2),
        y1 + 37,
        scale=0.78,
        color=(255, 255, 255),
        thickness=2
    )


def draw_pause_menu_screen():
    """
    Game-like Home Menu screen. It keeps current flowers visible behind the menu
    and uses the same rounded, colorful style as the rest of the game UI.
    """
    if pause_return_state == "stage2":
        frame = stage2_background_img.copy() if stage2_background_img is not None else background.copy()
        frame = draw_stage2_bush_on_pot(frame, "left", stage2_left_bush_stage)
        frame = draw_stage2_bush_on_pot(frame, "right", stage2_right_bush_stage)
    elif pause_return_state == "stage3":
        frame = draw_stage3_scene_base()
    elif pause_return_state == "stage4":
        frame = draw_stage4_scene_base()
    else:
        frame = background.copy()
        frame = draw_all_current_flowers(frame)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (245, 250, 238), -1)
    cv2.addWeighted(overlay, 0.36, frame, 0.64, 0, frame)

    # Soft playful sparkles around the menu card
    sparkle_points = [
        (270, 105), (365, 170), (945, 118), (1018, 198),
        (250, 580), (1020, 570), (380, 620), (905, 620)
    ]

    for index, (sx, sy) in enumerate(sparkle_points):
        pulse = 0.55 + 0.45 * math.sin(time.time() * 3.4 + index)
        radius = int(4 + 5 * pulse)
        cv2.circle(frame, (sx, sy), radius, (0, 225, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), radius + 5, (255, 255, 255), 1, cv2.LINE_AA)

    card_x1 = 300
    card_y1 = 75
    card_x2 = 980
    card_y2 = 650

    draw_transparent_rounded_rect(frame, card_x1 + 9, card_y1 + 11, card_x2 + 9, card_y2 + 11, (70, 80, 90), alpha=0.22, radius=38)
    draw_filled_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (255, 248, 220), radius=38)
    draw_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (40, 170, 110), radius=38, thickness=5)

    # Home badge at the top of the card
    badge_x1 = WIDTH // 2 - 42
    badge_y1 = card_y1 + 22
    badge_x2 = WIDTH // 2 + 42
    badge_y2 = card_y1 + 92
    draw_filled_rounded_rect(frame, badge_x1, badge_y1, badge_x2, badge_y2, (235, 250, 226), radius=28)
    draw_rounded_rect(frame, badge_x1, badge_y1, badge_x2, badge_y2, (45, 170, 115), radius=28, thickness=3)

    # Draw a larger house icon inside the badge
    cx = WIDTH // 2
    cv2.line(frame, (cx - 24, badge_y1 + 42), (cx, badge_y1 + 20), (40, 120, 85), 5, cv2.LINE_AA)
    cv2.line(frame, (cx, badge_y1 + 20), (cx + 24, badge_y1 + 42), (40, 120, 85), 5, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - 18, badge_y1 + 42), (cx + 18, badge_y1 + 62), (70, 165, 120), -1)
    cv2.rectangle(frame, (cx - 18, badge_y1 + 42), (cx + 18, badge_y1 + 62), (40, 120, 85), 2)
    cv2.rectangle(frame, (cx - 5, badge_y1 + 51), (cx + 5, badge_y1 + 62), (255, 248, 220), -1)

    draw_centered_text(frame, "HOME MENU", WIDTH // 2, 202, scale=1.20, color=(0, 145, 90), thickness=3)
    draw_centered_text(frame, "Your progress is safe", WIDTH // 2, 240, scale=0.62, color=(75, 95, 70), thickness=2)
    if pause_return_state == "tutorial":
        tutorial_score = 1 if tutorial_completed else 0
        draw_centered_text(frame, f"Training Stage: {tutorial_score} / 1", WIDTH // 2, 270, scale=0.58, color=(35, 115, 180), thickness=2)
    elif pause_return_state == "stage2":
        draw_centered_text(frame, f"Stage 2: {stage2_score} / {STAGE2_TOTAL_POTS}", WIDTH // 2, 270, scale=0.58, color=(35, 115, 180), thickness=2)
    elif pause_return_state == "stage3":
        draw_centered_text(frame, f"Stage 3: {stage3_score} / {STAGE3_TOTAL_POTS}", WIDTH // 2, 270, scale=0.58, color=(35, 115, 180), thickness=2)
    elif pause_return_state == "stage4":
        draw_centered_text(frame, f"Stage 4: {stage4_score} / {STAGE4_TOTAL_POTS}", WIDTH // 2, 270, scale=0.58, color=(35, 115, 180), thickness=2)
    else:
        draw_centered_text(frame, f"Score: {score} / {TOTAL_FLOWERS}", WIDTH // 2, 270, scale=0.58, color=(35, 115, 180), thickness=2)

    buttons = get_pause_menu_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    for button in buttons:
        draw_pause_menu_button(frame, button, hovered=(button["id"] == hover_button))

    draw_centered_text(frame, "Click a button", WIDTH // 2, 625, scale=0.55, color=(80, 80, 80), thickness=1)

    return frame


def shift_paused_game_timers(paused_seconds):
    """
    When the Home Menu or Recalibration screen pauses the game, time-based rain,
    tutorial delay, and return animations should not silently finish in the background.
    """
    global last_sun_move_time
    global sun_shining_start_time
    global rain_effect_start_time
    global cloud_activation_time
    global locked_rain_start_time
    global locked_stage3_pause_start_time
    global tutorial_stage3_complete_time
    global stage2_rain_start_time
    global stage2_stage3_pause_start_time
    global stage2_completion_time
    global stage3_rain_start_time
    global stage3_stage3_pause_start_time
    global stage3_completion_time
    global stage4_rain_start_time
    global stage4_stage3_pause_start_time
    global stage4_completion_time

    if paused_seconds <= 0:
        return

    if last_sun_move_time:
        last_sun_move_time += paused_seconds

    if sun_shining_start_time:
        sun_shining_start_time += paused_seconds

    if rain_effect_start_time:
        rain_effect_start_time += paused_seconds

    if cloud_activation_time:
        cloud_activation_time += paused_seconds

    if locked_rain_sequence_active and locked_rain_start_time:
        locked_rain_start_time += paused_seconds

    if locked_stage3_pause_active and locked_stage3_pause_start_time:
        locked_stage3_pause_start_time += paused_seconds

    if tutorial_stage3_complete_time is not None:
        tutorial_stage3_complete_time += paused_seconds

    if stage2_rain_sequence_active and stage2_rain_start_time:
        stage2_rain_start_time += paused_seconds

    if stage2_stage3_pause_active and stage2_stage3_pause_start_time:
        stage2_stage3_pause_start_time += paused_seconds

    if stage2_completion_time is not None:
        stage2_completion_time += paused_seconds
    if stage3_rain_sequence_active and stage3_rain_start_time:
        stage3_rain_start_time += paused_seconds
    if stage3_stage3_pause_active and stage3_stage3_pause_start_time:
        stage3_stage3_pause_start_time += paused_seconds
    if stage3_completion_time is not None:
        stage3_completion_time += paused_seconds
    if stage4_rain_sequence_active and stage4_rain_start_time:
        stage4_rain_start_time += paused_seconds
    if stage4_stage3_pause_active and stage4_stage3_pause_start_time:
        stage4_stage3_pause_start_time += paused_seconds
    if stage4_completion_time is not None:
        stage4_completion_time += paused_seconds


def enter_pause_menu(return_state=None):
    global game_state
    global pause_menu_enter_time
    global pause_return_state
    global mouse_left_clicked
    global locked_chin_tuck_last_update_time
    global tutorial_chin_tuck_last_update_time
    global stage3_chin_tuck_last_update_time
    global stage4_chin_tuck_last_update_time

    if return_state is None:
        return_state = game_state

    if return_state not in ["game", "tutorial", "stage2", "stage3", "stage4"]:
        return_state = "game"

    pause_return_state = return_state
    pause_current_session_clock()

    clear_all_movement_holds()
    locked_chin_tuck_last_update_time = None
    tutorial_chin_tuck_last_update_time = None
    stage2_chin_tuck_last_update_time = None
    pause_locked_shoulder_lift_progress()
    pause_locked_retraction_progress()
    pause_stage2_shoulder_progress()
    pause_stage2_retraction_progress()
    pause_stage3_shoulder_progress()
    pause_stage3_retraction_progress()
    stage3_chin_tuck_last_update_time = None
    pause_stage4_shoulder_progress()
    pause_stage4_retraction_progress()
    stage4_chin_tuck_last_update_time = None

    pause_menu_enter_time = time.time()
    mouse_left_clicked = False
    game_state = "pause_menu"


def resume_game_from_pause():
    global game_state
    global pause_menu_enter_time
    global pause_return_state
    global calibration_return_mode
    global mouse_left_clicked
    global locked_chin_tuck_last_update_time
    global tutorial_chin_tuck_last_update_time
    global stage3_chin_tuck_last_update_time
    global stage4_chin_tuck_last_update_time

    if pause_menu_enter_time is not None:
        shift_paused_game_timers(time.time() - pause_menu_enter_time)

    return_state = pause_return_state if pause_return_state in ["game", "tutorial", "stage2", "stage3", "stage4"] else "game"

    pause_menu_enter_time = None
    calibration_return_mode = "new_game"
    mouse_left_clicked = False

    clear_all_movement_holds()
    locked_chin_tuck_last_update_time = None
    tutorial_chin_tuck_last_update_time = None
    stage2_chin_tuck_last_update_time = None
    pause_locked_shoulder_lift_progress()
    pause_locked_retraction_progress()
    pause_stage2_shoulder_progress()
    pause_stage2_retraction_progress()
    pause_stage3_shoulder_progress()
    pause_stage3_retraction_progress()
    stage3_chin_tuck_last_update_time = None
    pause_stage4_shoulder_progress()
    pause_stage4_retraction_progress()
    stage4_chin_tuck_last_update_time = None

    resume_current_session_clock()
    game_state = return_state


def reset_calibration_only_keep_game_progress():
    """
    Recalibration from the Home Menu must NOT reset the game.
    It clears only saved calibration samples and live movement holds.
    Score, flowers, character position, active character and lock state stay untouched.
    """
    global neutral_pitch
    global neutral_yaw
    global neutral_roll
    global neutral_side_bend_angle
    global flexion_direction
    global flexion_threshold
    global extension_direction
    global extension_threshold
    global left_side_bend_direction
    global left_side_bend_threshold
    global right_side_bend_direction
    global right_side_bend_threshold
    global smoothed_pitch
    global smoothed_yaw
    global smoothed_roll
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    global chin_neutral_features
    global chin_target_features
    global chin_neutral_pitch
    global chin_neutral_yaw
    global chin_neutral_eye_roll
    global chin_neutral_face_width
    global smoothed_chin_eye_roll
    global chin_tuck_hold_start
    global stage_chin_hold_start
    global rain_chin_hold_start
    global stage_chin_last_seen_time
    global rain_chin_last_seen_time
    global rain_waiting_for_chin_release
    global shoulder_neutral_features
    global shoulder_target_features
    global shoulder_neutral_nose_y
    global shoulder_neutral_width
    global shoulder_neutral_angle
    global smoothed_shoulder_features
    global smoothed_shoulder_nose_y
    global smoothed_shoulder_width
    global smoothed_shoulder_angle
    global shoulder_hold_start
    global shoulder_release_start_time
    global shoulder_toggle_waiting_release
    global locked_chin_tuck_last_update_time

    neutral_pitch = None
    neutral_yaw = None
    neutral_roll = None
    neutral_side_bend_angle = None

    flexion_direction = None
    flexion_threshold = None
    extension_direction = None
    extension_threshold = None

    left_side_bend_direction = None
    left_side_bend_threshold = None
    right_side_bend_direction = None
    right_side_bend_threshold = None

    smoothed_pitch = None
    smoothed_yaw = None
    smoothed_roll = None

    flexion_hold_start = None
    extension_hold_start = None
    left_side_bend_hold_start = None
    right_side_bend_hold_start = None

    chin_neutral_features = None
    chin_target_features = None
    chin_neutral_pitch = None
    chin_neutral_yaw = None
    chin_neutral_eye_roll = None
    chin_neutral_face_width = None
    smoothed_chin_eye_roll = None
    clear_chin_histories()

    chin_tuck_hold_start = None
    stage_chin_hold_start = None
    rain_chin_hold_start = None
    stage_chin_last_seen_time = None
    rain_chin_last_seen_time = None
    rain_waiting_for_chin_release = False
    locked_chin_tuck_last_update_time = None

    shoulder_neutral_features = None
    shoulder_target_features = None
    shoulder_neutral_nose_y = None
    shoulder_neutral_width = None
    shoulder_neutral_angle = None
    smoothed_shoulder_features = None
    smoothed_shoulder_nose_y = None
    smoothed_shoulder_width = None
    smoothed_shoulder_angle = None
    clear_shoulder_histories()

    shoulder_hold_start = None
    shoulder_release_start_time = None
    shoulder_toggle_waiting_release = False
    pause_locked_shoulder_lift_progress()

    reset_retraction_calibration_state(clear_saved=True)
    pause_locked_retraction_progress()


def start_recalibration_from_pause_menu():
    global game_state
    global calibration_return_mode
    global mouse_left_clicked
    global selected_recalibration_target

    calibration_return_mode = "resume_game"
    selected_recalibration_target = None
    mouse_left_clicked = False
    game_state = "recalibrate_select"



def get_recalibrate_selection_buttons():
    w = 330
    h = 56
    gap_x = 55
    gap_y = 22
    left_x = int(WIDTH / 2 - w - gap_x / 2)
    right_x = int(WIDTH / 2 + gap_x / 2)
    y0 = 245

    items = [
        ("neutral", "Neutral Posture"),
        ("flexion", "Flexion"),
        ("extension", "Extension"),
        ("left_bend", "Left Side Bend"),
        ("right_bend", "Right Side Bend"),
        ("chin_tuck", "Chin Tuck"),
        ("shoulder_lift", "Scapular Elevation"),
        ("palm_retraction", "Scapular Retraction"),
    ]

    buttons = []
    for i, (button_id, label) in enumerate(items):
        col = i % 2
        row = i // 2
        x1 = left_x if col == 0 else right_x
        y1 = y0 + row * (h + gap_y)
        buttons.append({
            "id": f"recalibrate_{button_id}",
            "label": label,
            "rect": (x1, y1, x1 + w, y1 + h),
        })

    buttons.append({"id": "back_pause", "label": "Back", "rect": (1010, 615, 1190, 675)})
    return buttons


def draw_recalibrate_selection_screen():
    frame = draw_menu_background(overlay_alpha=0.25 if main_menu_background_img is not None else 0.35)

    draw_transparent_rounded_rect(frame, 230, 80, 1050, 680, (255, 252, 235), alpha=0.92, radius=40)
    draw_rounded_rect(frame, 230, 80, 1050, 680, (55, 170, 115), radius=40, thickness=5)

    draw_centered_text(frame, "Choose Movement", WIDTH // 2, 150, scale=1.12, color=(0, 145, 90), thickness=3)
    draw_centered_text(
        frame,
        "Select only the movement you want to recalibrate. Your game progress will stay safe.",
        WIDTH // 2,
        195,
        scale=0.52,
        color=(70, 90, 80),
        thickness=1
    )

    buttons = get_recalibrate_selection_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    for button in buttons:
        draw_main_menu_button(frame, button, hovered=(button["id"] == hover_button))

    return frame


def clear_selected_calibration_only(target):
    """
    Clears only the selected calibration target while keeping score, flowers,
    character position and the rest of the saved calibration data untouched.
    """
    global flexion_direction, flexion_threshold
    global extension_direction, extension_threshold
    global left_side_bend_direction, left_side_bend_threshold
    global right_side_bend_direction, right_side_bend_threshold
    global chin_target_features
    global shoulder_target_features
    global shoulder_hold_start, shoulder_release_start_time, shoulder_toggle_waiting_release
    global flexion_hold_start, extension_hold_start
    global left_side_bend_hold_start, right_side_bend_hold_start
    global stage_chin_hold_start, stage_chin_last_seen_time
    global rain_chin_hold_start, rain_chin_last_seen_time, rain_waiting_for_chin_release

    clear_all_movement_holds()
    pause_locked_shoulder_lift_progress()
    pause_locked_retraction_progress()
    pause_stage2_shoulder_progress()
    pause_stage2_retraction_progress()
    pause_stage3_shoulder_progress()
    pause_stage3_retraction_progress()
    pause_stage4_shoulder_progress()
    pause_stage4_retraction_progress()

    if target == "neutral":
        # Neutral posture is the base for every calibrated movement, so this option
        # intentionally starts a full recalibration while preserving game progress.
        reset_calibration_only_keep_game_progress()
        return

    if target == "flexion":
        flexion_direction = None
        flexion_threshold = None
        flexion_hold_start = None

    elif target == "extension":
        extension_direction = None
        extension_threshold = None
        extension_hold_start = None

    elif target == "left_bend":
        left_side_bend_direction = None
        left_side_bend_threshold = None
        left_side_bend_hold_start = None

    elif target == "right_bend":
        right_side_bend_direction = None
        right_side_bend_threshold = None
        right_side_bend_hold_start = None

    elif target == "chin_tuck":
        chin_target_features = None
        stage_chin_hold_start = None
        stage_chin_last_seen_time = None
        rain_chin_hold_start = None
        rain_chin_last_seen_time = None
        rain_waiting_for_chin_release = False
        clear_chin_histories()

    elif target == "shoulder_lift":
        shoulder_target_features = None
        shoulder_hold_start = None
        shoulder_release_start_time = None
        shoulder_toggle_waiting_release = False
        clear_shoulder_histories()
        # Scapular Retraction depends on the scapular elevation/retraction calibration flow, so it is
        # safely repeated after Scapular Elevation is updated.
        reset_retraction_calibration_state(clear_saved=True)

    elif target == "palm_retraction":
        reset_retraction_calibration_state(clear_saved=True)


def start_selected_recalibration_from_menu(target):
    global game_state
    global calibration_return_mode
    global mouse_left_clicked
    global selected_recalibration_target

    selected_recalibration_target = target
    calibration_return_mode = "resume_game"
    clear_selected_calibration_only(target)
    mouse_left_clicked = False
    game_state = "calibration"



def draw_win_button(frame, button, hovered=False):
    x1, y1, x2, y2 = button["rect"]

    if button["id"] == "quit":
        normal_fill = (85, 130, 235)
        hover_fill = (105, 155, 255)
        border = (255, 255, 255)
    else:
        normal_fill = (70, 165, 120)
        hover_fill = (95, 205, 150)
        border = (255, 255, 255)

    fill = hover_fill if hovered else normal_fill
    border_thickness = 4 if hovered else 2

    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=23)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=23, thickness=border_thickness)

    if hovered:
        cv2.circle(frame, (x1 + 25, int((y1 + y2) / 2)), 8, (255, 255, 255), -1)
        cv2.circle(frame, (x2 - 25, int((y1 + y2) / 2)), 8, (255, 255, 255), -1)

    draw_centered_text(
        frame,
        button["label"],
        int((x1 + x2) / 2),
        y1 + 36,
        scale=0.78,
        color=(255, 255, 255),
        thickness=2
    )


def draw_all_current_flowers(frame):
    flower_items = [
        (top_flower_stage, top_flower_stage1_img, top_flower_stage2_img, top_flower_stage3_img, TOP_POT_CENTER_X, TOP_POT_SOIL_Y),
        (bottom_flower_stage, bottom_flower_stage1_img, bottom_flower_stage2_img, bottom_flower_stage3_img, BOTTOM_POT_CENTER_X, BOTTOM_POT_SOIL_Y),
        (right_orchid_stage, right_orchid_stage1_img, right_orchid_stage2_img, right_orchid_stage3_img, RIGHT_ORCHID_POT_CENTER_X, RIGHT_ORCHID_POT_SOIL_Y),
        (south_east_bluebloom_stage, south_east_bluebloom_stage1_img, south_east_bluebloom_stage2_img, south_east_bluebloom_stage3_img, SOUTH_EAST_BLUEBLOOM_POT_CENTER_X, SOUTH_EAST_BLUEBLOOM_POT_SOIL_Y),
        (left_tulip_stage, left_tulip_stage1_img, left_tulip_stage2_img, left_tulip_stage3_img, LEFT_TULIP_POT_CENTER_X, LEFT_TULIP_POT_SOIL_Y),
        (south_west_peony_stage, south_west_peony_stage1_img, south_west_peony_stage2_img, south_west_peony_stage3_img, SOUTH_WEST_PEONY_POT_CENTER_X, SOUTH_WEST_PEONY_POT_SOIL_Y),
    ]

    for stage, img1, img2, img3, pot_x, pot_y in flower_items:
        flower_img = get_flower_asset(stage, img1, img2, img3)
        if flower_img is not None:
            frame = draw_flower_on_pot(frame, flower_img, pot_x, pot_y, 1.0)

    return frame



def draw_win_screen(frame):
    if current_stage_number == 4:
        frame = draw_stage4_scene_base()
    elif current_stage_number == 3:
        frame = draw_stage3_scene_base()
    elif current_stage_number == 2:
        frame = stage2_background_img.copy() if stage2_background_img is not None else background.copy()
        frame = draw_stage2_bush_on_pot(frame, "left", stage2_left_bush_stage)
        frame = draw_stage2_bush_on_pot(frame, "right", stage2_right_bush_stage)
    else:
        frame = draw_all_current_flowers(frame)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (255, 248, 225), -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
    sparkle_points = [(135,125),(215,240),(1080,135),(1140,260),(250,585),(1045,575),(365,135),(920,145),(180,425),(1100,430)]
    for index, (sx, sy) in enumerate(sparkle_points):
        pulse = 0.55 + 0.45 * math.sin(time.time() * 3.5 + index)
        radius = int(5 + 5 * pulse)
        cv2.circle(frame, (sx, sy), radius, (0, 230, 255), -1)
        cv2.circle(frame, (sx, sy), radius + 5, (255, 255, 255), 1)
    card_x1, card_y1, card_x2, card_y2 = 300, 75, 980, 650
    draw_transparent_rounded_rect(frame, card_x1 + 8, card_y1 + 10, card_x2 + 8, card_y2 + 10, (80, 90, 100), alpha=0.22, radius=35)
    draw_filled_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (255, 248, 220), radius=35)
    draw_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (40, 170, 110), radius=35, thickness=5)
    win_sun = cv2.resize(sun_frames[4] if len(sun_frames) > 4 else sun_frames[0], (150, 150), interpolation=cv2.INTER_AREA)
    frame = overlay_transparent(frame, win_sun, WIDTH // 2 - 75, 92)
    draw_centered_text(frame, "YOU WON!", WIDTH // 2, 285, scale=1.65, color=(0, 145, 90), thickness=4)
    if current_stage_number == 5:
        stage_line = "Stage 5 Complete - Main Garden"
        score_line = f"Final Score: {score} / {TOTAL_FLOWERS}"
    elif current_stage_number == 4:
        stage_line = "Stage 4 Complete - Winter Garden"
        score_line = f"Final Score: {stage4_score} / {STAGE4_TOTAL_POTS}"
    elif current_stage_number == 3:
        stage_line = "Stage 3 Complete - Autumn Garden"
        score_line = f"Final Score: {stage3_score} / {STAGE3_TOTAL_POTS}"
    elif current_stage_number == 2:
        stage_line = "Stage 2 Complete - Summer Pots"
        score_line = f"Final Score: {stage2_score} / {STAGE2_TOTAL_POTS}"
    else:
        stage_line = "Stage Complete"
        score_line = "Great work!"
    draw_centered_text(frame, stage_line, WIDTH // 2, 330, scale=0.72, color=(75, 95, 70), thickness=2)
    draw_centered_text(frame, score_line, WIDTH // 2, 375, scale=0.88, color=(35, 115, 180), thickness=2)
    if win_message != "":
        draw_centered_text(frame, win_message, WIDTH // 2, 405, scale=0.55, color=(90, 90, 90), thickness=1)
    buttons = get_win_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)
    for button in buttons:
        draw_win_button(frame, button, hovered=(button["id"] == hover_button))
    draw_centered_text(frame, "Click a button", WIDTH // 2, 628, scale=0.55, color=(80, 80, 80), thickness=1)
    return frame


def draw_placeholder_screen(title, subtitle):
    frame = background.copy()
    frame = draw_all_current_flowers(frame)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (245, 240, 220), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    draw_transparent_rounded_rect(frame, 300, 185, 980, 525, (255, 248, 220), alpha=0.92, radius=35)
    draw_rounded_rect(frame, 300, 185, 980, 525, (40, 170, 110), radius=35, thickness=5)

    draw_centered_text(frame, title, WIDTH // 2, 300, scale=1.25, color=(0, 145, 90), thickness=3)
    draw_centered_text(frame, subtitle, WIDTH // 2, 360, scale=0.70, color=(75, 95, 70), thickness=2)
    draw_centered_text(frame, "This screen will be designed in the next step.", WIDTH // 2, 415, scale=0.62, color=(80, 80, 80), thickness=2)
    draw_centered_text(frame, "Press R to recalibrate/play again or Q to quit", WIDTH // 2, 470, scale=0.58, color=(65, 65, 65), thickness=1)

    return frame




# -----------------------------
# Easy tutorial stage helpers
# -----------------------------
def reset_tutorial_stage_state():
    """
    Resets only the easy tutorial state. Calibration data is kept.
    """
    global tutorial_sun_current_x
    global tutorial_sun_current_y
    global tutorial_sun_target_x
    global tutorial_sun_target_y
    global tutorial_locked_to_center
    global tutorial_chin_tuck_total_time
    global tutorial_chin_tuck_last_update_time
    global tutorial_message
    global tutorial_completed
    global tutorial_stage3_complete_time
    global last_sun_move_time
    global active_character

    tutorial_sun_current_x = float(TUTORIAL_SUN_START_X)
    tutorial_sun_current_y = float(TUTORIAL_SUN_START_Y)
    tutorial_sun_target_x = float(TUTORIAL_SUN_START_X)
    tutorial_sun_target_y = float(TUTORIAL_SUN_START_Y)

    tutorial_locked_to_center = False
    tutorial_chin_tuck_total_time = 0.0
    tutorial_chin_tuck_last_update_time = None
    tutorial_message = "Move the sun to the center tree circle."
    tutorial_completed = False
    tutorial_stage3_complete_time = None

    active_character = "sun"
    last_sun_move_time = 0

    clear_all_movement_holds()
    reset_locked_chin_tuck_progress()
    reset_locked_shoulder_lift_progress()
    reset_locked_retraction_progress()
    reset_locked_rain_sequence()


def start_tutorial_stage_after_calibration():
    """
    Starts Stage 1: the easy one-tree training stage.
    """
    global game_state
    global mouse_left_clicked
    global win_message
    global current_stage_number

    reset_tutorial_stage_state()
    current_stage_number = 1
    win_message = ""
    mouse_left_clicked = False
    start_new_session_metrics("stage_1_play", stage_number=1)
    game_state = "tutorial"


def reset_main_game_play_state_keep_calibration():
    """
    Starts the six-pot main game after the tutorial without clearing calibration data.
    """
    global sun_current_x
    global sun_current_y
    global sun_target_x
    global sun_target_y
    global last_sun_move_time
    global sun_shining_start_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    global cloud_activation_time
    global active_character
    global score
    global top_flower_stage
    global bottom_flower_stage
    global right_orchid_stage
    global south_east_bluebloom_stage
    global left_tulip_stage
    global south_west_peony_stage
    global top_flower_animating
    global bottom_flower_animating
    global right_orchid_animating
    global south_east_bluebloom_animating
    global left_tulip_animating
    global south_west_peony_animating
    global top_flower_start_time
    global bottom_flower_start_time
    global right_orchid_start_time
    global south_east_bluebloom_start_time
    global left_tulip_start_time
    global south_west_peony_start_time
    global active_flower
    global character_locked_to_flower
    global locked_flower_key
    global reached_side_pots
    global game_finished
    global win_message

    sun_current_x = float(sun_x)
    sun_current_y = float(sun_y)
    sun_target_x = float(sun_x)
    sun_target_y = float(sun_y)

    last_sun_move_time = 0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    rain_effect_x = float(sun_x)
    rain_effect_y = float(sun_y)
    cloud_activation_time = 0.0
    active_character = "sun"

    score = 0
    top_flower_stage = 0
    bottom_flower_stage = 0
    right_orchid_stage = 0
    south_east_bluebloom_stage = 0
    left_tulip_stage = 0
    south_west_peony_stage = 0

    top_flower_animating = False
    bottom_flower_animating = False
    right_orchid_animating = False
    south_east_bluebloom_animating = False
    left_tulip_animating = False
    south_west_peony_animating = False

    top_flower_start_time = 0
    bottom_flower_start_time = 0
    right_orchid_start_time = 0
    south_east_bluebloom_start_time = 0
    left_tulip_start_time = 0
    south_west_peony_start_time = 0

    active_flower = None
    character_locked_to_flower = False
    locked_flower_key = None
    reached_side_pots = set()

    clear_all_movement_holds()
    reset_locked_chin_tuck_progress()
    reset_locked_shoulder_lift_progress()
    reset_locked_retraction_progress()
    reset_locked_rain_sequence()

    game_finished = False
    win_message = ""


def start_main_game_after_tutorial():
    global game_state
    global mouse_left_clicked
    global current_stage_number

    reset_main_game_play_state_keep_calibration()
    current_stage_number = 5
    mouse_left_clicked = False
    start_new_session_metrics("stage_5_play", stage_number=5)
    game_state = "game"


def get_tutorial_character_center():
    return int(tutorial_sun_target_x + SUN_SIZE / 2), int(tutorial_sun_target_y + SUN_SIZE / 2)


def tutorial_target_is_reached():
    cx, cy = get_tutorial_character_center()
    dx = cx - TUTORIAL_TREE_CENTER_X
    dy = cy - TUTORIAL_TREE_CENTER_Y
    return math.sqrt(dx * dx + dy * dy) <= TUTORIAL_TARGET_RADIUS


def clamp_tutorial_target():
    global tutorial_sun_target_x
    global tutorial_sun_target_y

    tutorial_sun_target_x = max(0, min(WIDTH - SUN_SIZE, tutorial_sun_target_x))
    tutorial_sun_target_y = max(0, min(HEIGHT - SUN_SIZE, tutorial_sun_target_y))


def update_tutorial_chin_tuck_progress(is_detected):
    global tutorial_chin_tuck_total_time
    global tutorial_chin_tuck_last_update_time

    now = time.time()

    if is_detected:
        if tutorial_chin_tuck_last_update_time is None:
            tutorial_chin_tuck_last_update_time = now
        else:
            elapsed = now - tutorial_chin_tuck_last_update_time
            if 0.0 <= elapsed <= 1.0:
                tutorial_chin_tuck_total_time += elapsed
            tutorial_chin_tuck_last_update_time = now
    else:
        tutorial_chin_tuck_last_update_time = None

    if tutorial_chin_tuck_total_time > TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:
        tutorial_chin_tuck_total_time = TUTORIAL_CHIN_REQUIRED_TOTAL_TIME

    return tutorial_chin_tuck_total_time


def get_tutorial_tree_stage():
    if not tutorial_locked_to_center:
        return 0

    progress = tutorial_chin_tuck_total_time / max(TUTORIAL_CHIN_REQUIRED_TOTAL_TIME, 0.1)

    if progress >= 1.0:
        return 3
    if progress >= 0.50:
        return 2
    return 1


def get_tutorial_tree_asset(stage):
    if stage == 1:
        return tree_stage1_img
    if stage == 2:
        return tree_stage2_img
    if stage >= 3:
        return tree_stage3_img
    return None


def draw_tutorial_tree(frame, stage):
    if stage <= 0:
        return frame

    asset = get_tutorial_tree_asset(stage)

    if asset is None:
        # Fallback drawn tree if the PNG asset is missing.
        trunk_h = 80 + stage * 35
        trunk_w = 18 + stage * 8
        base_x = TUTORIAL_TREE_CENTER_X + TUTORIAL_TREE_DRAW_OFFSET_X
        base_y = TUTORIAL_TREE_BASE_Y
        cv2.rectangle(frame, (base_x - trunk_w // 2, base_y - trunk_h), (base_x + trunk_w // 2, base_y), (95, 65, 35), -1)
        cv2.circle(frame, (base_x, base_y - trunk_h), 45 + stage * 18, (75, 170, 85), -1, cv2.LINE_AA)
        if stage >= 3:
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                fx = int(base_x + math.cos(rad) * 55)
                fy = int(base_y - trunk_h + math.sin(rad) * 42)
                cv2.circle(frame, (fx, fy), 9, (190, 130, 245), -1, cv2.LINE_AA)
        return frame

    desired_h = {1: 190, 2: 255, 3: 320}.get(stage, 280)
    img_h, img_w = asset.shape[:2]
    scale = desired_h / max(img_h, 1)
    desired_w = max(1, int(img_w * scale))
    desired_h = max(1, int(img_h * scale))

    resized = cv2.resize(asset, (desired_w, desired_h), interpolation=cv2.INTER_AREA)

    x = int(TUTORIAL_TREE_CENTER_X + TUTORIAL_TREE_DRAW_OFFSET_X - desired_w / 2)
    y = int(TUTORIAL_TREE_BASE_Y - desired_h)

    return overlay_transparent(frame, resized, x, y)


def process_tutorial_stage(
    current_pitch,
    current_yaw,
    current_roll,
    current_side_bend_angle,
    current_chin_features
):
    """
    Easy tutorial:
    1) Move the sun from the lower-right area to the center ring.
    2) Lock the sun above the center.
    3) Hold Chin Tuck for 10 cumulative seconds to grow the tree.
    """
    global tutorial_sun_target_x
    global tutorial_sun_target_y
    global tutorial_sun_current_x
    global tutorial_sun_current_y
    global tutorial_locked_to_center
    global tutorial_message
    global tutorial_completed
    global tutorial_stage3_complete_time
    global game_state
    global win_message
    global last_sun_move_time
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start

    now = time.time()

    if tutorial_completed:
        return "Tutorial complete."

    if not tutorial_locked_to_center:
        tutorial_message = "Move to the center tree circle."

        if tutorial_target_is_reached():
            tutorial_locked_to_center = True
            tutorial_sun_target_x = float(TUTORIAL_TREE_CENTER_X - SUN_SIZE / 2)
            tutorial_sun_target_y = float(TUTORIAL_SUN_LOCK_Y)
            tutorial_message = "Great! Now do Chin Tuck and hold it for 10 seconds."
            clear_all_movement_holds()

        else:
            # First check left/right side-bend movement.
            side_moved = False

            if (
                current_side_bend_angle is not None and
                neutral_side_bend_angle is not None and
                left_side_bend_direction is not None and
                left_side_bend_threshold is not None and
                right_side_bend_direction is not None and
                right_side_bend_threshold is not None
            ):
                side_bend_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
                left_amount = left_side_bend_direction * side_bend_delta
                right_amount = right_side_bend_direction * side_bend_delta

                if left_amount >= left_side_bend_threshold:
                    if left_side_bend_hold_start is None:
                        left_side_bend_hold_start = now
                    if now - left_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
                        tutorial_sun_target_x -= TUTORIAL_MOVE_DISTANCE
                        last_sun_move_time = now
                        left_side_bend_hold_start = None
                        right_side_bend_hold_start = None
                        side_moved = True
                        tutorial_message = "Good! Keep moving toward the center circle."
                else:
                    left_side_bend_hold_start = None

                if not side_moved:
                    if right_amount >= right_side_bend_threshold:
                        if right_side_bend_hold_start is None:
                            right_side_bend_hold_start = now
                        if now - right_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
                            tutorial_sun_target_x += TUTORIAL_MOVE_DISTANCE
                            last_sun_move_time = now
                            left_side_bend_hold_start = None
                            right_side_bend_hold_start = None
                            side_moved = True
                            tutorial_message = "Good! Keep moving toward the center circle."
                    else:
                        right_side_bend_hold_start = None

            # Then check up/down movement if no side-bend movement was used.
            if not side_moved and current_pitch is not None and neutral_pitch is not None and flexion_direction is not None and extension_direction is not None:
                yaw_delta = abs(angle_diff(current_yaw, neutral_yaw)) if current_yaw is not None and neutral_yaw is not None else 0.0

                if yaw_delta <= MAX_ALLOWED_YAW_CHANGE:
                    pitch_delta = angle_diff(current_pitch, neutral_pitch)
                    flexion_amount = flexion_direction * pitch_delta
                    extension_amount = extension_direction * pitch_delta

                    if flexion_threshold is not None and flexion_amount >= flexion_threshold:
                        if flexion_hold_start is None:
                            flexion_hold_start = now
                        if now - flexion_hold_start >= FLEXION_REQUIRED_HOLD_TIME:
                            tutorial_sun_target_y += TUTORIAL_MOVE_DISTANCE
                            flexion_hold_start = None
                            extension_hold_start = None
                            tutorial_message = "Good! Flexion moves the character down toward the center circle."
                    else:
                        flexion_hold_start = None

                    if extension_threshold is not None and extension_amount >= extension_threshold:
                        if extension_hold_start is None:
                            extension_hold_start = now
                        if now - extension_hold_start >= EXTENSION_REQUIRED_HOLD_TIME:
                            tutorial_sun_target_y -= TUTORIAL_MOVE_DISTANCE
                            extension_hold_start = None
                            flexion_hold_start = None
                            tutorial_message = "Good! Extension moves the character up toward the center circle."
                    else:
                        extension_hold_start = None
                else:
                    tutorial_message = "Keep your face forward while moving."

            clamp_tutorial_target()

            if tutorial_target_is_reached():
                tutorial_locked_to_center = True
                tutorial_sun_target_x = float(TUTORIAL_TREE_CENTER_X - SUN_SIZE / 2)
                tutorial_sun_target_y = float(TUTORIAL_SUN_LOCK_Y)
                tutorial_message = "Great! Now do Chin Tuck and hold it for 10 seconds."
                clear_all_movement_holds()

    else:
        # Chin Tuck growth section.
        if (
            current_chin_features is not None and
            chin_neutral_features is not None and
            chin_target_features is not None
        ):
            is_chin_tuck, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                current_chin_features,
                chin_neutral_features,
                chin_target_features,
                current_pitch,
                neutral_pitch,
                current_yaw,
                neutral_yaw,
                current_roll,
                neutral_roll
            )

            total_time = update_tutorial_chin_tuck_progress(is_chin_tuck)

            if is_chin_tuck:
                tutorial_message = f"Chin Tuck: {total_time:.1f}s / {TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
            else:
                tutorial_message = f"Hold Chin Tuck for 10 seconds. Progress: {total_time:.1f}s / {TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:.1f}s"

            if total_time >= TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:
                if tutorial_stage3_complete_time is None:
                    tutorial_stage3_complete_time = now
                    clear_all_movement_holds()

                elapsed_after_stage3 = now - tutorial_stage3_complete_time
                remaining_to_win = max(0.0, TUTORIAL_STAGE3_TO_WIN_DELAY - elapsed_after_stage3)

                if elapsed_after_stage3 >= TUTORIAL_STAGE3_TO_WIN_DELAY:
                    tutorial_completed = True
                    mark_stage_completed(1)
                    finalize_session_save("completed")
                    win_message = "Stage 1 complete. Easy Tree is fully grown."
                    game_state = "tutorial_win"
                    clear_all_movement_holds()
                    return "Stage 1 complete. You won!"

                tutorial_message = f"Tree is fully grown! Win menu in {remaining_to_win:.1f}s"
        else:
            tutorial_chin_tuck_last_update_time = None
            tutorial_message = "Keep your face visible, then do Chin Tuck."

    tutorial_sun_current_x += (tutorial_sun_target_x - tutorial_sun_current_x) * 0.22
    tutorial_sun_current_y += (tutorial_sun_target_y - tutorial_sun_current_y) * 0.22

    return tutorial_message


def draw_tutorial_instruction_card(frame):
    if not tutorial_locked_to_center:
        title = "Training Stage"
        line1 = "Move to the center tree circle"
        line2 = "Use the calibrated neck movements to guide the sun."
    else:
        if tutorial_chin_tuck_total_time >= TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:
            title = "Great Job!"
            line1 = "The tree is fully grown"
            if tutorial_stage3_complete_time is not None:
                remaining = max(0.0, TUTORIAL_STAGE3_TO_WIN_DELAY - (time.time() - tutorial_stage3_complete_time))
                line2 = f"Win menu will open in {remaining:.1f}s"
            else:
                line2 = "Get ready for the win menu."
        else:
            title = "Chin Tuck Training"
            line1 = "Now do Chin Tuck"
            line2 = f"Hold for 10 seconds: {tutorial_chin_tuck_total_time:.1f} / {TUTORIAL_CHIN_REQUIRED_TOTAL_TIME:.1f}s"

    draw_transparent_rounded_rect(frame, 245, 28, 1035, 132, (255, 248, 220), alpha=0.92, radius=30)
    draw_rounded_rect(frame, 245, 28, 1035, 132, (55, 170, 115), radius=30, thickness=4)

    draw_centered_text(frame, title, WIDTH // 2, 66, scale=0.88, color=(0, 145, 90), thickness=3)
    draw_centered_text(frame, line1, WIDTH // 2, 96, scale=0.56, color=(50, 75, 70), thickness=2)
    draw_centered_text(frame, line2, WIDTH // 2, 120, scale=0.43, color=(75, 85, 90), thickness=1)

    if tutorial_locked_to_center:
        bar_x1, bar_y1, bar_x2, bar_y2 = 410, 146, 870, 164
        draw_filled_rounded_rect(frame, bar_x1, bar_y1, bar_x2, bar_y2, (225, 232, 230), radius=9)
        ratio = tutorial_chin_tuck_total_time / max(TUTORIAL_CHIN_REQUIRED_TOTAL_TIME, 0.1)
        fill_x2 = int(bar_x1 + (bar_x2 - bar_x1) * max(0.0, min(1.0, ratio)))
        if fill_x2 > bar_x1:
            draw_filled_rounded_rect(frame, bar_x1, bar_y1, fill_x2, bar_y2, (70, 185, 115), radius=9)
        draw_rounded_rect(frame, bar_x1, bar_y1, bar_x2, bar_y2, (65, 145, 90), radius=9, thickness=2)


def draw_tutorial_screen(current_sun_frame=None):
    if tutorial_background_img is not None:
        frame = tutorial_background_img.copy()
    else:
        frame = background.copy()

    # Readable warm overlay, very light so the tutorial background remains visible.
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (255, 248, 225), -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    # Center tree growth target / protective circle.
    pulse = 0.5 + 0.5 * math.sin(time.time() * 3.0)
    circle_color = (0, 220, 255) if not tutorial_locked_to_center else (80, 210, 120)
    cv2.circle(frame, (TUTORIAL_TREE_CENTER_X, TUTORIAL_TREE_CENTER_Y), int(TUTORIAL_TARGET_RADIUS + 7 * pulse), circle_color, 4, cv2.LINE_AA)
    cv2.circle(frame, (TUTORIAL_TREE_CENTER_X, TUTORIAL_TREE_CENTER_Y), 11, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (TUTORIAL_TREE_CENTER_X, TUTORIAL_TREE_CENTER_Y), 7, circle_color, -1, cv2.LINE_AA)

    # Draw tree after the user reaches the center.
    tree_stage = get_tutorial_tree_stage()
    frame = draw_tutorial_tree(frame, tree_stage)

    # Draw the sun character.
    if current_sun_frame is not None:
        sun_frame = current_sun_frame
    elif len(sun_frames) > 0:
        sun_frame = sun_frames[0]
    else:
        sun_frame = None

    if sun_frame is not None:
        if tutorial_locked_to_center:
            frame = draw_sun_glow(frame, int(tutorial_sun_current_x), int(tutorial_sun_current_y), SUN_SIZE)
        frame = overlay_transparent(frame, sun_frame, int(tutorial_sun_current_x), int(tutorial_sun_current_y))

    draw_tutorial_instruction_card(frame)

    # Home button for the tutorial stage, same as the main six-pot game.
    home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
    frame = draw_home_icon_button(frame, hovered=home_hovered)

    return frame


def get_tutorial_win_buttons():
    button_w = 430
    button_h = 54
    center_x = WIDTH // 2
    x1 = int(center_x - button_w / 2)
    x2 = x1 + button_w

    y1 = 420
    gap = 68

    return [
        {"id": "next_level", "label": "Next Stage", "rect": (x1, y1, x2, y1 + button_h)},
        {"id": "main_menu", "label": "Back to Main Menu", "rect": (x1, y1 + gap, x2, y1 + gap + button_h)},
        {"id": "quit", "label": "Quit", "rect": (x1, y1 + 2 * gap, x2, y1 + 2 * gap + button_h)},
    ]


def draw_tutorial_win_screen():
    if tutorial_background_img is not None:
        frame = tutorial_background_img.copy()
    else:
        frame = background.copy()

    frame = draw_tutorial_tree(frame, 3)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (255, 248, 225), -1)
    cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)

    sparkle_points = [
        (135, 125), (215, 240), (1080, 135), (1140, 260),
        (250, 585), (1045, 575), (365, 135), (920, 145),
        (180, 425), (1100, 430)
    ]

    for index, (sx, sy) in enumerate(sparkle_points):
        pulse = 0.55 + 0.45 * math.sin(time.time() * 3.5 + index)
        radius = int(5 + 5 * pulse)
        cv2.circle(frame, (sx, sy), radius, (0, 230, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (sx, sy), radius + 5, (255, 255, 255), 1, cv2.LINE_AA)

    card_x1 = 300
    card_y1 = 75
    card_x2 = 980
    card_y2 = 650

    draw_transparent_rounded_rect(frame, card_x1 + 8, card_y1 + 10, card_x2 + 8, card_y2 + 10, (80, 90, 100), alpha=0.22, radius=35)
    draw_filled_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (255, 248, 220), radius=35)
    draw_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (40, 170, 110), radius=35, thickness=5)

    if len(sun_frames) > 4:
        win_sun = cv2.resize(sun_frames[4], (150, 150), interpolation=cv2.INTER_AREA)
    else:
        win_sun = cv2.resize(sun_frames[0], (150, 150), interpolation=cv2.INTER_AREA)

    frame = overlay_transparent(frame, win_sun, WIDTH // 2 - 75, 92)

    draw_centered_text(frame, "YOU WON!", WIDTH // 2, 285, scale=1.65, color=(0, 145, 90), thickness=4)
    draw_centered_text(frame, "Stage 1 Complete - Easy Tree", WIDTH // 2, 330, scale=0.75, color=(75, 95, 70), thickness=2)
    draw_centered_text(frame, "The tree is fully grown", WIDTH // 2, 375, scale=0.70, color=(35, 115, 180), thickness=2)

    buttons = get_tutorial_win_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    for button in buttons:
        draw_win_button(frame, button, hovered=(button["id"] == hover_button))

    draw_centered_text(frame, "Click a button", WIDTH // 2, 628, scale=0.55, color=(80, 80, 80), thickness=1)

    return frame



# -----------------------------
# Main Menu / Profile / Progress / Settings helpers
# -----------------------------
def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def load_json_file(path, default_value):
    try:
        if not os.path.exists(path):
            return default_value
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_value


def save_json_file(path, data):
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_profile_id(name):
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    cleaned = cleaned.strip("_")
    if cleaned == "":
        cleaned = "guest"
    return cleaned[:32] + "_" + datetime.now().strftime("%Y%m%d%H%M%S")


def load_profiles():
    data = load_json_file(PROFILES_PATH, {"profiles": []})
    if "profiles" not in data or not isinstance(data["profiles"], list):
        data = {"profiles": []}
    return data


def load_active_profile():
    profiles_data = load_profiles()
    active_data = load_json_file(ACTIVE_PROFILE_PATH, {})
    active_id = active_data.get("profile_id")

    for profile in profiles_data["profiles"]:
        if profile.get("profile_id") == active_id:
            return profile

    if len(profiles_data["profiles"]) > 0:
        profile = profiles_data["profiles"][-1]
        save_json_file(ACTIVE_PROFILE_PATH, {"profile_id": profile.get("profile_id")})
        return profile

    return None


def save_profile(profile):
    profiles_data = load_profiles()
    existing_index = None

    for i, item in enumerate(profiles_data["profiles"]):
        if item.get("profile_id") == profile.get("profile_id"):
            existing_index = i
            break

    if existing_index is None:
        profiles_data["profiles"].append(profile)
    else:
        profiles_data["profiles"][existing_index] = profile

    save_json_file(PROFILES_PATH, profiles_data)
    save_json_file(ACTIVE_PROFILE_PATH, {"profile_id": profile.get("profile_id")})
    return profile


def default_profile_form_fields(profile=None):
    profile = profile or {}
    return [
        {"key": "name", "label": "Name", "value": str(profile.get("name", ""))},
        {"key": "age", "label": "Age", "value": str(profile.get("age", ""))},
        {"key": "gender", "label": "Gender", "value": str(profile.get("gender", ""))},
        {"key": "daily_sitting_hours", "label": "Daily computer sitting hours", "value": str(profile.get("daily_sitting_hours", ""))},
        {"key": "height_cm", "label": "Height (cm)", "value": str(profile.get("height_cm", ""))},
        {"key": "weight_kg", "label": "Weight (kg)", "value": str(profile.get("weight_kg", ""))},
        {"key": "neck_pain_level", "label": "Neck/shoulder pain level 0-10", "value": str(profile.get("neck_pain_level", ""))},
    ]


def build_profile_from_form():
    global current_profile

    profile = dict(current_profile) if current_profile is not None else {}
    if profile.get("profile_id") is None:
        profile["profile_id"] = normalize_profile_id(profile_form_fields[0]["value"])
        profile["created_at"] = datetime.now().isoformat(timespec="seconds")

    for field in profile_form_fields:
        profile[field["key"]] = field["value"].strip()

    profile["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return profile


def draw_input_field(frame, x1, y1, x2, y2, label, value, active=False):
    fill = (255, 255, 255) if active else (245, 250, 247)
    border = (65, 165, 230) if active else (105, 180, 130)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=18)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=18, thickness=3 if active else 2)
    draw_ui_text(frame, label, x1 + 18, y1 + 25, scale=0.45, color=(50, 95, 75), thickness=1)

    shown = value if value != "" else "Type here..."
    shown_color = (35, 45, 55) if value != "" else (150, 155, 160)
    if len(shown) > 38:
        shown = shown[:35] + "..."
    draw_ui_text(frame, shown, x1 + 18, y1 + 56, scale=0.58, color=shown_color, thickness=2)


def get_profile_buttons():
    return [
        {"id": "save_profile", "label": "Save Profile", "rect": (760, 610, 975, 670)},
        {"id": "back_main", "label": "Back", "rect": (1000, 610, 1190, 670)},
    ]


def draw_profile_screen():
    frame = draw_fantasy_page_frame(
        "Create Profile",
        "Save personal information once. It will be used for your future progress history."
    )

    # Profile card
    draw_transparent_rounded_rect(frame, 150, 150, 1130, 590, (255, 255, 245), alpha=0.93, radius=32)
    draw_rounded_rect(frame, 150, 150, 1130, 590, (70, 170, 115), radius=32, thickness=3)

    x_left = 190
    x_right = 665
    y_start = 185
    field_w = 425
    field_h = 68
    gap = 18

    for i, field in enumerate(profile_form_fields):
        col_x = x_left if i < 4 else x_right
        row_i = i if i < 4 else i - 4
        y = y_start + row_i * (field_h + gap)
        draw_input_field(
            frame,
            col_x,
            y,
            col_x + field_w,
            y + field_h,
            field["label"],
            field["value"],
            active=(i == profile_active_field_index)
        )

    # Clean saved-message area. No long keyboard guide text here.
    if profile_message != "":
        msg_color = (45, 140, 85) if "saved" in profile_message.lower() else (70, 90, 220)
        draw_filled_rounded_rect(frame, 410, 520, 870, 562, (235, 250, 230), radius=20)
        draw_rounded_rect(frame, 410, 520, 870, 562, (75, 175, 115), radius=20, thickness=2)
        draw_centered_text(frame, profile_message, WIDTH // 2, 548, scale=0.58, color=msg_color, thickness=2)

    buttons = get_profile_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)
    for button in buttons:
        draw_main_menu_button(frame, button, hovered=(button["id"] == hover_button))

    return frame


def handle_profile_keyboard(key):
    global profile_active_field_index
    global profile_message
    global current_profile
    global profile_form_fields
    global game_state

    if key == 255:
        return

    if key == 27:
        game_state = "main_menu"
        profile_message = ""
        return

    if key in [9, 13]:
        profile_active_field_index = (profile_active_field_index + 1) % len(profile_form_fields)
        return

    if key in [8, 127]:
        current_value = profile_form_fields[profile_active_field_index]["value"]
        profile_form_fields[profile_active_field_index]["value"] = current_value[:-1]
        return

    if 32 <= key <= 126:
        ch = chr(key)
        if len(profile_form_fields[profile_active_field_index]["value"]) < 40:
            profile_form_fields[profile_active_field_index]["value"] += ch


def profile_click_handler(clicked_button):
    global current_profile
    global profile_message
    global game_state

    if clicked_button == "save_profile":
        candidate = build_profile_from_form()
        if candidate.get("name", "").strip() == "":
            profile_message = "Name is required."
            return
        current_profile = save_profile(candidate)
        profile_message = "Information saved."

    elif clicked_button == "back_main":
        game_state = "main_menu"
        profile_message = ""


def get_music_files():
    """
    Returns up to 6 music files from the music folder.
    Uses absolute paths so pygame can load files more reliably on Windows.
    """
    if not os.path.isdir(MUSIC_DIR):
        return []

    files = []
    for filename in sorted(os.listdir(MUSIC_DIR)):
        if filename.lower().endswith(SUPPORTED_MUSIC_EXTENSIONS):
            files.append(os.path.abspath(os.path.join(MUSIC_DIR, filename)))

    return files[:6]


def load_settings_data():
    data = load_json_file(SETTINGS_PATH, {})
    return {
        "volume": float(data.get("volume", 0.45)),
        "selected_music_index": data.get("selected_music_index", None),
    }


def save_settings_data():
    save_json_file(SETTINGS_PATH, {
        "volume": music_volume,
        "selected_music_index": selected_music_index,
    })


def init_music_system():
    global music_available
    global music_error_message

    if pygame is None:
        music_available = False
        music_error_message = "pygame is not installed. Run: pip install pygame"
        return False

    try:
        # More reliable mixer setup for Windows laptops.
        try:
            pygame.mixer.pre_init(44100, -16, 2, 512)
        except Exception:
            pass

        pygame.init()
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        music_available = True
        music_error_message = ""
        return True

    except Exception as exc:
        music_available = False
        music_error_message = f"Audio init error: {str(exc)[:70]}"
        return False


def play_music_index(index, allow_retry=True):
    global current_music_index
    global selected_music_index
    global music_error_message

    if not music_available:
        music_error_message = "Music system is OFF."
        return False

    if len(music_files) == 0:
        music_error_message = "No music files found in the music folder."
        return False

    if index is None or index < 0 or index >= len(music_files):
        music_error_message = "Selected music number is not available."
        return False

    try:
        music_path = os.path.abspath(music_files[index])

        if not os.path.exists(music_path):
            music_error_message = f"Music file not found: Music {index + 1}"
            return False

        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(max(0.0, min(1.0, music_volume)))
        pygame.mixer.music.play(loops=0)

        current_music_index = index
        selected_music_index = index
        music_error_message = f"Playing Music {index + 1}"
        save_settings_data()
        return True

    except Exception as exc:
        music_error_message = f"Cannot play Music {index + 1}: {str(exc)[:70]}"

        # Try another track automatically. This helps when one MP3 format is not supported.
        if allow_retry and len(music_files) > 1:
            for step in range(1, len(music_files)):
                next_index = (index + step) % len(music_files)
                if next_index != index and play_music_index(next_index, allow_retry=False):
                    return True

        return False


def start_random_music():
    if len(music_files) == 0:
        return False
    return play_music_index(random.randint(0, len(music_files) - 1))


def play_next_music():
    if len(music_files) == 0:
        return False
    if current_music_index is None:
        return start_random_music()
    return play_music_index((current_music_index + 1) % len(music_files))


def update_music_playback():
    global music_error_message

    if not music_available or len(music_files) == 0:
        return

    try:
        # If the current track finished, automatically start the next one.
        if current_music_index is not None and not pygame.mixer.music.get_busy():
            play_next_music()
        elif current_music_index is None:
            start_random_music()
    except Exception as exc:
        music_error_message = f"Music playback error: {str(exc)[:70]}"


def set_music_volume(new_volume):
    global music_volume
    music_volume = max(0.0, min(1.0, float(new_volume)))
    if music_available:
        try:
            pygame.mixer.music.set_volume(music_volume)
        except Exception:
            pass
    save_settings_data()


def get_settings_buttons():
    buttons = [
        {"id": "volume_down", "label": "-", "rect": (345, 220, 405, 280)},
        {"id": "volume_up", "label": "+", "rect": (815, 220, 875, 280)},
        {"id": "back_main", "label": "Back", "rect": (1010, 615, 1190, 675)},
    ]

    # Six centered music cards. They are lower now so the title never overlaps them.
    w = 190
    h = 70
    gap_x = 46
    gap_y = 26
    total_w = 3 * w + 2 * gap_x
    x0 = int((WIDTH - total_w) / 2)
    y0 = 420

    for i in range(6):
        col = i % 3
        row = i // 3
        buttons.append({
            "id": f"music_{i}",
            "label": f"Music {i + 1}",
            "rect": (
                x0 + col * (w + gap_x),
                y0 + row * (h + gap_y),
                x0 + col * (w + gap_x) + w,
                y0 + row * (h + gap_y) + h
            ),
        })

    return buttons


def draw_settings_screen():
    frame = draw_fantasy_page_frame(
        "Settings",
        "Adjust background music volume and choose one of the six game tracks."
    )

    # Main settings card
    draw_transparent_rounded_rect(frame, 160, 155, 1120, 635, (255, 255, 245), alpha=0.94, radius=34)
    draw_rounded_rect(frame, 160, 155, 1120, 635, (70, 170, 115), radius=34, thickness=3)

    # Sound section
    draw_filled_rounded_rect(frame, 215, 185, 1065, 300, (85, 170, 120), radius=28)
    draw_rounded_rect(frame, 215, 185, 1065, 300, (255, 255, 255), radius=28, thickness=2)

    # Keep the sound area clean: icon + controls + percentage only.
    draw_speaker_icon(frame, 285, 242, size=58, muted=(music_volume <= 0.01))
    draw_volume_bar(frame, 425, 248, 790, 272, music_volume)
    draw_ui_text(frame, f"{int(music_volume * 100)}%", 910, 245, scale=0.80, color=(255, 255, 255), thickness=3)

    # Music status
    if music_available and len(music_files) > 0:
        if current_music_index is not None:
            status = f"Playing Music {current_music_index + 1}"
        else:
            status = "Music system is ON"
        status_color = (35, 135, 80)
    elif pygame is None:
        status = "Music system is OFF | Install pygame"
        status_color = (85, 95, 220)
    elif len(music_files) == 0:
        status = "No music files found. Put MP3/WAV/OGG files inside the music folder."
        status_color = (85, 95, 220)
    else:
        status = "Music system is OFF"
        status_color = (85, 95, 220)

    if music_error_message != "":
        status = music_error_message

    draw_centered_text(frame, status, WIDTH // 2, 342, scale=0.50, color=status_color, thickness=2)

    # Music selection title. It is higher than the cards, with clean spacing.
    draw_centered_text(frame, "Choose Music", WIDTH // 2, 385, scale=0.72, color=(40, 120, 80), thickness=2)

    buttons = get_settings_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    for button in buttons:
        if button["id"] in ["volume_down", "volume_up", "back_main"]:
            draw_small_game_button(
                frame,
                button,
                hovered=(button["id"] == hover_button),
                selected=False,
                disabled=False,
                text_scale=0.72 if button["id"] != "back_main" else 0.58
            )
            continue

        if button["id"].startswith("music_"):
            index = int(button["id"].split("_")[1])
            disabled = index >= len(music_files)
            selected = index == current_music_index

            x1, y1, x2, y2 = button["rect"]
            if disabled:
                fill = (205, 210, 212)
                border = (155, 165, 168)
                text_color = (105, 112, 115)
            elif selected:
                fill = (245, 170, 70)
                border = (255, 255, 255)
                text_color = (255, 255, 255)
            else:
                fill = (70, 170, 115)
                border = (255, 255, 255)
                text_color = (255, 255, 255)

            if button["id"] == hover_button and not disabled:
                fill = (95, 210, 145) if not selected else (255, 190, 85)

            draw_transparent_rounded_rect(frame, x1 + 5, y1 + 7, x2 + 5, y2 + 7, (70, 80, 90), alpha=0.18, radius=24)
            draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=24)
            draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=24, thickness=4 if selected else 2)

            draw_centered_text(
                frame,
                button["label"],
                int((x1 + x2) / 2),
                y1 + 45,
                scale=0.62,
                color=text_color,
                thickness=2
            )

            if selected:
                cv2.circle(frame, (x2 - 22, y1 + 22), 8, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (x2 - 22, y1 + 22), 4, (245, 170, 70), -1, cv2.LINE_AA)

    return frame

def settings_click_handler(clicked_button):
    global game_state

    if clicked_button == "volume_down":
        set_music_volume(music_volume - 0.10)
    elif clicked_button == "volume_up":
        set_music_volume(music_volume + 0.10)
    elif clicked_button == "back_main":
        game_state = "main_menu"
    elif clicked_button is not None and clicked_button.startswith("music_"):
        index = int(clicked_button.split("_")[1])
        if index < len(music_files):
            play_music_index(index)


def get_main_menu_buttons():
    button_w = 390
    x1 = int(WIDTH / 2 - button_w / 2)
    x2 = x1 + button_w
    y1 = 255
    h = 58
    gap = 70
    return [
        {"id": "start_game", "label": "Start Game", "rect": (x1, y1, x2, y1 + h)},
        {"id": "create_profile", "label": "Create Profile", "rect": (x1, y1 + gap, x2, y1 + gap + h)},
        {"id": "progress", "label": "Analysis", "rect": (x1, y1 + 2 * gap, x2, y1 + 2 * gap + h)},
        {"id": "settings", "label": "Settings", "rect": (x1, y1 + 3 * gap, x2, y1 + 3 * gap + h)},
        {"id": "exit", "label": "Exit Game", "rect": (x1, y1 + 4 * gap, x2, y1 + 4 * gap + h)},
    ]


def draw_main_menu_button(frame, button, hovered=False, selected=False):
    x1, y1, x2, y2 = button["rect"]

    if button["id"] in ["exit", "quit"]:
        fill = (85, 120, 235)
        hover_fill = (110, 145, 255)
    elif selected:
        fill = (235, 165, 65)
        hover_fill = (250, 185, 80)
    else:
        fill = (65, 170, 115)
        hover_fill = (92, 210, 145)

    color = hover_fill if hovered else fill
    draw_transparent_rounded_rect(frame, x1 + 6, y1 + 7, x2 + 6, y2 + 7, (70, 80, 90), alpha=0.20, radius=24)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, color, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), radius=24, thickness=4 if hovered or selected else 2)

    draw_centered_text(
        frame,
        button["label"],
        int((x1 + x2) / 2),
        y1 + 38,
        scale=0.76,
        color=(255, 255, 255),
        thickness=2
    )


def draw_flower_decoration(frame, x, y, scale=1.0):
    stem_h = int(55 * scale)
    cv2.line(frame, (x, y), (x, y - stem_h), (55, 145, 70), max(2, int(4 * scale)), cv2.LINE_AA)
    center = (x, y - stem_h)
    petal_r = max(4, int(11 * scale))
    for ang in range(0, 360, 60):
        rad = math.radians(ang)
        px = int(center[0] + math.cos(rad) * petal_r)
        py = int(center[1] + math.sin(rad) * petal_r)
        cv2.circle(frame, (px, py), petal_r, (90, 120, 245), -1, cv2.LINE_AA)
    cv2.circle(frame, center, max(4, int(8 * scale)), (0, 210, 255), -1, cv2.LINE_AA)


def draw_river_and_flowers_panel(frame, x1, y1, x2, y2):
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (224, 248, 230), radius=32)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 170, 120), radius=32, thickness=4)

    # River
    pts = np.array([
        [x1 + 45, y1 + 80], [x1 + 120, y1 + 145], [x1 + 70, y1 + 230],
        [x1 + 155, y1 + 315], [x1 + 95, y1 + 430], [x1 + 190, y2 - 45],
        [x1 + 235, y2 - 45], [x1 + 145, y1 + 420], [x1 + 210, y1 + 315],
        [x1 + 130, y1 + 230], [x1 + 185, y1 + 150], [x1 + 100, y1 + 80]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (235, 185, 90), cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 3, cv2.LINE_AA)

    for i in range(5):
        yy = y1 + 120 + i * 95
        cv2.line(frame, (x1 + 70, yy), (x1 + 160, yy + 18), (255, 255, 255), 2, cv2.LINE_AA)

    for fx, fy, sc in [
        (x1 + 55, y2 - 60, 0.85),
        (x1 + 235, y2 - 70, 0.90),
        (x1 + 65, y1 + 290, 0.72),
        (x1 + 230, y1 + 210, 0.75),
        (x1 + 80, y1 + 135, 0.65),
    ]:
        draw_flower_decoration(frame, fx, fy, sc)

    draw_centered_text(frame, "Garden Path", int((x1 + x2) / 2), y1 + 52, scale=0.68, color=(35, 125, 80), thickness=2)


def draw_tree_sun_cloud_panel(frame, x1, y1, x2, y2, current_sun_frame=None, current_cloud_frame=None):
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (231, 247, 255), radius=32)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 170, 120), radius=32, thickness=4)

    # Hills
    cv2.ellipse(frame, (x1 + 95, y2 - 40), (150, 85), 0, 180, 360, (130, 210, 120), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - 70, y2 - 35), (155, 95), 0, 180, 360, (115, 200, 110), -1, cv2.LINE_AA)

    # Trees
    for tx, ty, sc in [(x1 + 80, y2 - 100, 1.0), (x1 + 205, y2 - 82, 0.78), (x2 - 85, y2 - 108, 0.92)]:
        trunk_w = int(22 * sc)
        trunk_h = int(95 * sc)
        cv2.rectangle(frame, (tx - trunk_w // 2, ty - trunk_h), (tx + trunk_w // 2, ty), (95, 65, 35), -1)
        cv2.circle(frame, (tx, ty - trunk_h), int(48 * sc), (45, 150, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, (tx - int(35 * sc), ty - trunk_h + int(25 * sc)), int(38 * sc), (55, 170, 90), -1, cv2.LINE_AA)
        cv2.circle(frame, (tx + int(35 * sc), ty - trunk_h + int(25 * sc)), int(38 * sc), (60, 180, 95), -1, cv2.LINE_AA)

    # Sun / cloud from game assets if available
    if current_sun_frame is not None:
        small_sun = cv2.resize(current_sun_frame, (118, 118), interpolation=cv2.INTER_AREA)
        overlay_transparent(frame, small_sun, x1 + 34, y1 + 42)
    else:
        cv2.circle(frame, (x1 + 92, y1 + 100), 48, (0, 225, 255), -1, cv2.LINE_AA)

    if current_cloud_frame is not None:
        small_cloud = cv2.resize(current_cloud_frame, (120, 120), interpolation=cv2.INTER_AREA)
        overlay_transparent(frame, small_cloud, x2 - 155, y1 + 92)
    else:
        cv2.ellipse(frame, (x2 - 95, y1 + 150), (60, 35), 0, 0, 360, (245, 245, 245), -1, cv2.LINE_AA)

    draw_centered_text(frame, "Bloom World", int((x1 + x2) / 2), y1 + 52, scale=0.68, color=(35, 125, 80), thickness=2)


def draw_main_menu_screen(current_sun_frame=None, current_cloud_frame=None):
    # If main_menu_background.png exists, this screen becomes a fantasy garden menu.
    # Otherwise it falls back to the drawn garden panels.
    frame = draw_menu_background(overlay_alpha=0.12 if main_menu_background_img is not None else 0.50)

    # Soft vignette for readability
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (20, 90, 70), -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

    # Decorative left/right side panels only when no custom image is provided.
    if main_menu_background_img is None:
        draw_river_and_flowers_panel(frame, 35, 35, 300, 685)
        draw_tree_sun_cloud_panel(frame, 825, 35, 1245, 685, current_sun_frame=current_sun_frame, current_cloud_frame=current_cloud_frame)

    # Title card
    title_x1, title_y1, title_x2, title_y2 = 330, 42, 950, 178
    draw_transparent_rounded_rect(frame, title_x1 + 8, title_y1 + 9, title_x2 + 8, title_y2 + 9, (50, 60, 60), alpha=0.22, radius=36)
    draw_transparent_rounded_rect(frame, title_x1, title_y1, title_x2, title_y2, (255, 248, 220), alpha=0.92, radius=36)
    draw_rounded_rect(frame, title_x1, title_y1, title_x2, title_y2, (65, 170, 115), radius=36, thickness=5)

    # Game characters belong to the title panel, not around the buttons.
    if current_sun_frame is not None:
        title_sun = cv2.resize(current_sun_frame, (76, 76), interpolation=cv2.INTER_AREA)
        overlay_transparent(frame, title_sun, title_x1 + 35, title_y1 + 31)

    if current_cloud_frame is not None:
        title_cloud = cv2.resize(current_cloud_frame, (82, 82), interpolation=cv2.INTER_AREA)
        overlay_transparent(frame, title_cloud, title_x2 - 115, title_y1 + 28)

    draw_centered_text(frame, "Bloom Motion", WIDTH // 2 + 4, 121, scale=1.55, color=(40, 70, 65), thickness=7)
    draw_centered_text(frame, "Bloom Motion", WIDTH // 2, 118, scale=1.55, color=(0, 145, 90), thickness=4)
    draw_centered_text(frame, "Neck & Shoulder Rehab Game", WIDTH // 2, 154, scale=0.52, color=(75, 95, 80), thickness=1)

    # Main button card
    card_x1, card_y1, card_x2, card_y2 = 365, 205, 915, 665
    draw_transparent_rounded_rect(frame, card_x1 + 8, card_y1 + 10, card_x2 + 8, card_y2 + 10, (45, 60, 60), alpha=0.24, radius=36)
    draw_transparent_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (255, 252, 235), alpha=0.92, radius=36)
    draw_rounded_rect(frame, card_x1, card_y1, card_x2, card_y2, (55, 170, 115), radius=36, thickness=5)

    if current_profile is not None:
        profile_text = f"Active profile: {current_profile.get('name', 'User')}"
    else:
        profile_text = "No active profile yet"
    draw_centered_text(frame, profile_text, WIDTH // 2, card_y1 + 34, scale=0.50, color=(55, 85, 95), thickness=1)

    # Use the existing button rectangles but shift them slightly to the new center card.
    buttons = get_main_menu_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    for button in buttons:
        draw_main_menu_button(frame, button, hovered=(button["id"] == hover_button))

    draw_centered_text(frame, "Click a button to continue", WIDTH // 2, card_y2 - 20, scale=0.46, color=(80, 90, 90), thickness=1)

    return frame



# -----------------------------
# Stage selection helpers
# -----------------------------
STAGE_DEFINITIONS = [
    {
        "number": 1,
        "title": "Easy Tree",
        "subtitle": "Learn the basic movements",
        "implemented": True,
    },
    {
        "number": 2,
        "title": "Summer Pots",
        "subtitle": "Grow two bushes with guided movements",
        "implemented": True,
    },
    {
        "number": 3,
        "title": "Autumn Garden",
        "subtitle": "Grow three autumn plants",
        "implemented": True,
    },
    {
        "number": 4,
        "title": "Winter Garden",
        "subtitle": "Thaw and grow four winter flowers",
        "implemented": True,
    },
    {
        "number": 5,
        "title": "Main Garden",
        "subtitle": "Six flower pots challenge",
        "implemented": True,
    },
]


def get_default_stage_progress():
    return {
        "unlocked_stage": 1,
        "completed_stages": [],
        "last_selected_stage": None,
    }


def normalize_stage_progress(progress):
    if not isinstance(progress, dict):
        progress = get_default_stage_progress()

    try:
        unlocked_stage = int(progress.get("unlocked_stage", 1))
    except Exception:
        unlocked_stage = 1

    unlocked_stage = max(1, min(5, unlocked_stage))

    completed = progress.get("completed_stages", [])
    if not isinstance(completed, list):
        completed = []

    normalized_completed = []
    for item in completed:
        try:
            value = int(item)
        except Exception:
            continue
        if 1 <= value <= 5 and value not in normalized_completed:
            normalized_completed.append(value)

    last_selected = progress.get("last_selected_stage", None)
    try:
        last_selected = int(last_selected) if last_selected is not None else None
    except Exception:
        last_selected = None

    if last_selected is not None and not (1 <= last_selected <= 5):
        last_selected = None

    return {
        "unlocked_stage": unlocked_stage,
        "completed_stages": sorted(normalized_completed),
        "last_selected_stage": last_selected,
    }


def get_stage_progress():
    if current_profile is not None:
        return normalize_stage_progress(current_profile.get("stage_progress", {}))
    return normalize_stage_progress(guest_stage_progress)


def save_stage_progress(progress):
    global current_profile
    global guest_stage_progress

    progress = normalize_stage_progress(progress)

    if current_profile is not None:
        current_profile["stage_progress"] = progress
        current_profile = save_profile(current_profile)
    else:
        guest_stage_progress = progress

    return progress


def mark_stage_completed(stage_number):
    try:
        stage_number = int(stage_number)
    except Exception:
        return

    if not (1 <= stage_number <= 5):
        return

    progress = get_stage_progress()
    completed = list(progress.get("completed_stages", []))

    if stage_number not in completed:
        completed.append(stage_number)

    progress["completed_stages"] = sorted(completed)
    progress["unlocked_stage"] = max(int(progress.get("unlocked_stage", 1)), min(5, stage_number + 1))
    save_stage_progress(progress)


def get_stage_definition(stage_number):
    for item in STAGE_DEFINITIONS:
        if item["number"] == stage_number:
            return item
    return None


def stage_is_unlocked(stage_number):
    # Temporary development mode: show every stage as unlocked in the selector.
    # Implemented stages are playable immediately; unimplemented stages still
    # remain disabled and are labeled Coming Soon.
    if TEMP_UNLOCK_ALL_STAGES:
        return get_stage_definition(stage_number) is not None

    progress = get_stage_progress()
    return int(stage_number) <= int(progress.get("unlocked_stage", 1))


def stage_is_completed(stage_number):
    progress = get_stage_progress()
    return int(stage_number) in progress.get("completed_stages", [])


def stage_is_playable(stage_number):
    stage_def = get_stage_definition(stage_number)
    return (
        stage_def is not None and
        stage_def.get("implemented", False) and
        stage_is_unlocked(stage_number)
    )


def get_stage_status_text(stage_number):
    stage_def = get_stage_definition(stage_number)
    if stage_def is None:
        return "Locked"

    if stage_is_completed(stage_number):
        return "Completed"

    if not stage_is_unlocked(stage_number):
        if stage_number == 1:
            return "Locked"
        return f"Complete Stage {stage_number - 1}"

    if not stage_def.get("implemented", False):
        return "Coming Soon"

    return "Available"


def get_stage_button_label(stage_number):
    if stage_is_completed(stage_number):
        return "Replay"
    if stage_is_playable(stage_number):
        return "Play Stage"
    if stage_is_unlocked(stage_number):
        return "Coming Soon"
    return "Locked"


def draw_lock_icon(frame, cx, cy, scale=1.0, color=(80, 80, 80)):
    s = float(scale)
    shackle_w = int(54 * s)
    shackle_h = int(42 * s)
    body_w = int(66 * s)
    body_h = int(48 * s)
    body_x1 = int(cx - body_w / 2)
    body_y1 = int(cy - body_h / 2 + 8 * s)
    body_x2 = body_x1 + body_w
    body_y2 = body_y1 + body_h

    cv2.ellipse(frame, (cx, body_y1 + 2), (shackle_w // 2, shackle_h // 2), 180, 0, 180, color, max(3, int(5 * s)), cv2.LINE_AA)
    draw_filled_rounded_rect(frame, body_x1, body_y1, body_x2, body_y2, (246, 244, 230), radius=max(8, int(12 * s)))
    draw_rounded_rect(frame, body_x1, body_y1, body_x2, body_y2, color, radius=max(8, int(12 * s)), thickness=max(2, int(3 * s)))
    cv2.circle(frame, (cx, int(body_y1 + 22 * s)), max(4, int(6 * s)), color, -1, cv2.LINE_AA)
    cv2.line(frame, (cx, int(body_y1 + 24 * s)), (cx, int(body_y1 + 36 * s)), color, max(2, int(3 * s)), cv2.LINE_AA)


def draw_check_icon(frame, cx, cy, radius=14):
    cv2.circle(frame, (cx, cy), radius, (75, 175, 80), -1, cv2.LINE_AA)
    cv2.line(frame, (cx - int(radius * 0.45), cy), (cx - int(radius * 0.12), cy + int(radius * 0.35)), (255, 255, 255), 4, cv2.LINE_AA)
    cv2.line(frame, (cx - int(radius * 0.12), cy + int(radius * 0.35)), (cx + int(radius * 0.55), cy - int(radius * 0.45)), (255, 255, 255), 4, cv2.LINE_AA)


def get_stage_accent_color(stage_number, completed=False):
    """
    Pastel colors for the stage selector. Completed stages always become green;
    locked / coming-soon stages keep their own soft color instead of dull gray.
    """
    if completed:
        return (66, 176, 92)

    palette = {
        1: (74, 178, 86),    # green
        2: (240, 142, 95),   # peach / flower patch
        3: (88, 166, 210),   # sky blue / trellis
        4: (176, 132, 210),  # soft purple / blossom tree
        5: (210, 168, 58),   # warm gold / main garden
    }
    return palette.get(int(stage_number), (90, 170, 110))


def blend_color(color, bg=(255, 255, 255), alpha=0.18):
    """
    Returns a light pastel version of a BGR color.
    """
    return tuple(int(color[i] * alpha + bg[i] * (1.0 - alpha)) for i in range(3))


def draw_stage_image_fit(frame, image, x1, y1, x2, y2, padding=8):
    """
    Fits a BGR/BGRA image inside the preview box without letting it overflow.
    """
    if image is None:
        return False

    inner_x1 = x1 + padding
    inner_y1 = y1 + padding
    inner_x2 = x2 - padding
    inner_y2 = y2 - padding

    box_w = max(1, inner_x2 - inner_x1)
    box_h = max(1, inner_y2 - inner_y1)
    img_h, img_w = image.shape[:2]

    scale = min(box_w / max(img_w, 1), box_h / max(img_h, 1))
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    px = inner_x1 + int((box_w - new_w) / 2)
    py = inner_y1 + int((box_h - new_h) / 2)

    # Extra safety: never write outside the frame or preview area.
    px = max(inner_x1, min(px, inner_x2 - new_w))
    py = max(inner_y1, min(py, inner_y2 - new_h))

    if len(resized.shape) == 3 and resized.shape[2] == 4:
        overlay_transparent(frame, resized, px, py)
    else:
        frame[py:py + new_h, px:px + new_w] = resized

    return True


def get_stage_preview_image(stage_number):
    if stage_number == 1:
        return tree_stage3_img
    if stage_number == 2:
        return stage2_preview_img
    if stage_number == 3:
        return stage3_preview_img
    if stage_number == 4:
        return stage4_preview_img
    return None


def draw_stage5_flower_grid(frame, x1, y1, x2, y2):
    flower_assets = [
        top_flower_stage3_img,
        bottom_flower_stage3_img,
        right_orchid_stage3_img,
        south_east_bluebloom_stage3_img,
        left_tulip_stage3_img,
        south_west_peony_stage3_img,
    ]

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cell_w = box_w / 3.0
    cell_h = box_h / 2.0
    icon_size = int(min(cell_w * 0.74, cell_h * 0.74, 54))
    icon_size = max(34, icon_size)

    for index, img in enumerate(flower_assets):
        col = index % 3
        row = index // 3
        cx = int(x1 + cell_w * (col + 0.5))
        cy = int(y1 + cell_h * (row + 0.5))
        px = int(cx - icon_size / 2)
        py = int(cy - icon_size / 2)

        if img is not None:
            small = cv2.resize(img, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
            overlay_transparent(frame, small, px, py)
        else:
            cv2.circle(frame, (cx, cy - 4), int(icon_size * 0.30), (120, 95, 220), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (px + int(icon_size * 0.28), py + int(icon_size * 0.66)), (px + int(icon_size * 0.72), py + int(icon_size * 0.95)), (150, 105, 70), -1, cv2.LINE_AA)


def draw_stage_preview(frame, stage_number, x1, y1, x2, y2, locked=False):
    completed = stage_is_completed(stage_number)
    accent = get_stage_accent_color(stage_number, completed=completed)
    fill = blend_color(accent, alpha=0.16)
    border = accent if not locked else blend_color(accent, bg=(210, 225, 220), alpha=0.65)

    draw_transparent_rounded_rect(frame, x1 + 3, y1 + 4, x2 + 3, y2 + 4, (70, 80, 85), alpha=0.12, radius=20)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=20)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=20, thickness=2)

    if stage_number in [1, 2, 3, 4]:
        image = get_stage_preview_image(stage_number)
        if not draw_stage_image_fit(frame, image, x1, y1, x2, y2, padding=9):
            # Clean drawn fallback if the external PNG was not saved yet.
            cx = int((x1 + x2) / 2)
            base_y = y2 - 22
            cv2.rectangle(frame, (cx - 12, base_y - 75), (cx + 12, base_y), (100, 70, 40), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, base_y - 94), 42, accent, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx - 35, base_y - 70), 34, blend_color(accent, alpha=0.75), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx + 35, base_y - 70), 34, blend_color(accent, alpha=0.75), -1, cv2.LINE_AA)

    elif stage_number == 5:
        draw_stage5_flower_grid(frame, x1 + 10, y1 + 10, x2 - 10, y2 - 10)

    if locked:
        overlay = frame.copy()
        draw_filled_rounded_rect(overlay, x1, y1, x2, y2, (248, 246, 230), radius=20)
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)
        draw_lock_icon(frame, int((x1 + x2) / 2), int((y1 + y2) / 2), scale=0.66, color=(72, 88, 82))

def get_stage_select_buttons():
    buttons = []
    card_w = 196
    card_h = 408
    gap = 14
    start_x = int((WIDTH - (5 * card_w + 4 * gap)) / 2)
    y1 = 220

    for stage_def in STAGE_DEFINITIONS:
        index = stage_def["number"] - 1
        x1 = start_x + index * (card_w + gap)
        buttons.append({
            "id": f"stage_{stage_def['number']}",
            "label": stage_def["title"],
            "rect": (x1, y1, x1 + card_w, y1 + card_h),
        })

    buttons.append({"id": "back_main", "label": "Back to Menu", "rect": (995, 128, 1190, 182)})
    return buttons

def draw_stage_card(frame, stage_def, rect, hovered=False):
    x1, y1, x2, y2 = rect
    stage_number = stage_def["number"]
    completed = stage_is_completed(stage_number)
    implemented = stage_def.get("implemented", False)
    playable = stage_is_playable(stage_number)
    unlocked = stage_is_unlocked(stage_number)
    locked_visual = not unlocked

    accent = get_stage_accent_color(stage_number, completed=completed)
    fill = blend_color(accent, alpha=0.12)
    border = accent
    title_color = (35, 95, 60)

    if completed:
        fill = (232, 255, 232)
        border = (66, 176, 92)
    elif playable:
        fill = (250, 255, 235)
        border = accent
    elif unlocked and not implemented:
        fill = blend_color(accent, bg=(255, 252, 236), alpha=0.17)
        border = accent
        title_color = (72, 80, 75)
    else:
        fill = blend_color(accent, bg=(255, 252, 236), alpha=0.11)
        border = blend_color(accent, bg=(170, 185, 180), alpha=0.70)
        title_color = (74, 78, 72)

    if hovered and playable:
        fill = blend_color(accent, bg=(255, 255, 235), alpha=0.24)
        border = (50, 205, 115)

    draw_transparent_rounded_rect(frame, x1 + 4, y1 + 6, x2 + 4, y2 + 6, (65, 75, 80), alpha=0.14, radius=26)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=26)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=26, thickness=4 if (playable or completed or hovered) else 3)

    badge_fill = (66, 176, 92) if completed else accent
    draw_filled_rounded_rect(frame, x1 + 43, y1 + 15, x2 - 43, y1 + 53, badge_fill, radius=14)
    draw_rounded_rect(frame, x1 + 43, y1 + 15, x2 - 43, y1 + 53, (255, 255, 255), radius=14, thickness=2)
    draw_centered_text(frame, f"Stage {stage_number}", int((x1 + x2) / 2), y1 + 40, scale=0.52, color=(255, 255, 255), thickness=2)

    preview_x1, preview_y1 = x1 + 16, y1 + 66
    preview_x2, preview_y2 = x2 - 16, y1 + 218
    draw_stage_preview(frame, stage_number, preview_x1, preview_y1, preview_x2, preview_y2, locked=locked_visual)

    draw_centered_text(frame, stage_def["title"], int((x1 + x2) / 2), y1 + 260, scale=0.51, color=title_color, thickness=2)

    # Keep subtitles short and safely inside the card.
    subtitle_lines = wrap_text_lines(stage_def["subtitle"], x2 - x1 - 34, scale=0.31, thickness=1)
    sy = y1 + 292
    for line in subtitle_lines[:2]:
        draw_centered_text(frame, line, int((x1 + x2) / 2), sy, scale=0.31, color=(78, 84, 78), thickness=1)
        sy += 18

    status = get_stage_status_text(stage_number)
    status_y = y1 + 342
    if completed:
        draw_check_icon(frame, x1 + 43, status_y - 6, radius=13)
        draw_ui_text(frame, status, x1 + 63, status_y, scale=0.40, color=(58, 150, 70), thickness=2)
    elif playable:
        draw_check_icon(frame, x1 + 41, status_y - 6, radius=12)
        draw_ui_text(frame, status, x1 + 60, status_y, scale=0.40, color=(58, 150, 70), thickness=2)
    elif unlocked and not implemented:
        cv2.circle(frame, (x1 + 44, status_y - 7), 11, accent, -1, cv2.LINE_AA)
        draw_ui_text(frame, status, x1 + 63, status_y, scale=0.39, color=(82, 86, 78), thickness=2)
    else:
        draw_lock_icon(frame, x1 + 43, status_y - 11, scale=0.25, color=(82, 92, 86))
        # Keep the status short so it does not overflow.
        short_status = "Locked"
        draw_ui_text(frame, short_status, x1 + 63, status_y, scale=0.40, color=(82, 92, 86), thickness=2)

    button_label = get_stage_button_label(stage_number)
    btn_x1, btn_y1, btn_x2, btn_y2 = x1 + 20, y2 - 54, x2 - 20, y2 - 15
    if playable:
        btn_fill = (55, 160, 70) if completed else accent
        if hovered:
            btn_fill = (60, 205, 105)
        draw_filled_rounded_rect(frame, btn_x1, btn_y1, btn_x2, btn_y2, btn_fill, radius=17)
        draw_rounded_rect(frame, btn_x1, btn_y1, btn_x2, btn_y2, (255, 255, 255), radius=17, thickness=2)
        draw_centered_text(frame, button_label, int((btn_x1 + btn_x2) / 2), btn_y1 + 26, scale=0.42, color=(255, 255, 255), thickness=2)
    else:
        disabled_fill = blend_color(accent, bg=(235, 244, 238), alpha=0.18)
        draw_filled_rounded_rect(frame, btn_x1, btn_y1, btn_x2, btn_y2, disabled_fill, radius=17)
        draw_rounded_rect(frame, btn_x1, btn_y1, btn_x2, btn_y2, blend_color(accent, bg=(180, 195, 190), alpha=0.65), radius=17, thickness=1)
        draw_centered_text(frame, button_label, int((btn_x1 + btn_x2) / 2), btn_y1 + 26, scale=0.36, color=(82, 88, 82), thickness=1)

def draw_stage_select_screen():
    if stage_select_background_img is not None:
        frame = stage_select_background_img.copy()
    else:
        frame = draw_menu_background(overlay_alpha=0.12 if main_menu_background_img is not None else 0.22)

    # Title above the main panel, as requested.
    draw_centered_text(frame, "Choose Stage", WIDTH // 2 + 4, 83, scale=1.38, color=(35, 65, 45), thickness=7)
    draw_centered_text(frame, "Choose Stage", WIDTH // 2, 80, scale=1.38, color=(35, 135, 70), thickness=4)

    # Main glass panel: a little lower and wider-safe so no card leaves the frame.
    panel_x1, panel_y1 = 72, 106
    panel_x2, panel_y2 = 1208, 688
    draw_transparent_rounded_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2, (255, 252, 235), alpha=0.90, radius=40)
    draw_rounded_rect(frame, panel_x1, panel_y1, panel_x2, panel_y2, (70, 150, 82), radius=40, thickness=5)

    profile_name = current_profile.get("name", "Guest") if current_profile is not None else "Guest"
    # Removed the old profile sticker; keep only a clean small text pill.
    draw_filled_rounded_rect(frame, 112, 130, 322, 166, (244, 255, 238), radius=17)
    draw_rounded_rect(frame, 112, 130, 322, 166, (125, 190, 120), radius=17, thickness=2)
    draw_ui_text(frame, f"Profile: {profile_name}", 132, 154, scale=0.45, color=(55, 65, 55), thickness=2)

    buttons = get_stage_select_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)
    back_button = buttons[-1]
    draw_small_game_button(frame, back_button, hovered=(hover_button == "back_main"), selected=False, disabled=False, text_scale=0.48)

    # No fixed explanatory lines under the title; this keeps the page breathable.
    if stage_select_message != "":
        draw_filled_rounded_rect(frame, 430, 135, 850, 166, (255, 248, 220), radius=14)
        draw_rounded_rect(frame, 430, 135, 850, 166, (230, 160, 115), radius=14, thickness=1)
        draw_centered_text(frame, stage_select_message, WIDTH // 2, 156, scale=0.34, color=(80, 95, 210), thickness=1)

    for stage_def, button in zip(STAGE_DEFINITIONS, buttons[:-1]):
        hovered = hover_button == button["id"]
        draw_stage_card(frame, stage_def, button["rect"], hovered=hovered)

    return frame

def start_new_game_from_main_menu():
    global game_state
    global mouse_left_clicked
    global stage_select_message
    global selected_stage_number
    global current_stage_number

    finalize_session_save("open_stage_select")
    selected_stage_number = None
    current_stage_number = None
    stage_select_message = ""
    mouse_left_clicked = False
    game_state = "level_select"


def start_stage_calibration_from_select(stage_number):
    global selected_stage_number
    global current_stage_number
    global calibration_return_mode
    global stage_select_message
    global selected_recalibration_target

    if not stage_is_unlocked(stage_number):
        stage_select_message = f"Complete Stage {stage_number - 1} first."
        return

    stage_def = get_stage_definition(stage_number)
    if stage_def is None:
        stage_select_message = "This stage is not available."
        return

    if not stage_def.get("implemented", False):
        stage_select_message = f"Stage {stage_number} will be designed next."
        return

    finalize_session_save(f"stage_{stage_number}_selected")
    selected_stage_number = stage_number
    current_stage_number = None

    reset_full_game_to_initial_state()
    selected_stage_number = stage_number
    selected_recalibration_target = None
    calibration_return_mode = "stage_start"
    stage_select_message = ""
    start_new_session_metrics(f"stage_{stage_number}_calibration")


def start_selected_stage_after_calibration():
    global selected_stage_number
    global stage_select_message

    if selected_stage_number == 1:
        start_tutorial_stage_after_calibration()
        return

    if selected_stage_number == 2:
        start_stage2_after_calibration()
        return

    if selected_stage_number == 3:
        start_stage3_after_calibration()
        return

    if selected_stage_number == 4:
        start_stage4_after_calibration()
        return

    if selected_stage_number == 5:
        start_main_game_after_tutorial()
        return

    stage_select_message = "This stage will be designed next."
    selected_stage_number = None


def start_stage_direct_without_calibration(stage_number):
    """
    Starts a stage from a completed-stage Next button without clearing calibration.
    Full calibration is intentionally required only when the user selects a stage
    from the level-select screen.
    """
    global selected_stage_number
    global stage_select_message
    global calibration_return_mode
    global pause_menu_enter_time
    global selected_recalibration_target
    global win_message

    stage_def = get_stage_definition(stage_number)
    if stage_def is None:
        stage_select_message = "This stage is not available."
        return False

    if not stage_def.get("implemented", False):
        stage_select_message = f"Stage {stage_number} is unlocked but will be designed next."
        return False

    selected_stage_number = stage_number
    stage_select_message = ""
    calibration_return_mode = "new_game"
    pause_menu_enter_time = None
    selected_recalibration_target = None
    win_message = ""

    finalize_session_save(f"stage_{stage_number}_direct_start")
    start_new_session_metrics(f"stage_{stage_number}_direct_start")

    if stage_number == 1:
        start_tutorial_stage_after_calibration()
        return True

    if stage_number == 2:
        start_stage2_after_calibration()
        return True

    if stage_number == 3:
        start_stage3_after_calibration()
        return True

    if stage_number == 4:
        start_stage4_after_calibration()
        return True

    if stage_number == 5:
        start_main_game_after_tutorial()
        return True

    stage_select_message = f"Stage {stage_number} is unlocked but will be designed next."
    return False


def start_next_stage_from_completion():
    """
    Next/Replay button after a completed stage.
    It does NOT restart full calibration. Full calibration is restarted only when
    a stage card is clicked from the level-select screen.
    """
    global game_state
    global stage_select_message

    next_stage = None

    if current_stage_number is not None and current_stage_number < 5:
        next_stage = current_stage_number + 1

    if next_stage is None:
        replay_stage = current_stage_number if current_stage_number is not None else 1
        if not start_stage_direct_without_calibration(replay_stage):
            game_state = "level_select"
        return

    if stage_is_playable(next_stage):
        if not start_stage_direct_without_calibration(next_stage):
            game_state = "level_select"
        return

    stage_select_message = f"Stage {next_stage} is unlocked but will be designed next."
    game_state = "level_select"


# -----------------------------
# Stage 2 - Summer Garden helpers
# -----------------------------
def get_stage2_pot_name(pot_key):
    names = {
        "left": "Left summer pot",
        "right": "Right summer pot",
    }
    return names.get(pot_key, "Summer pot")


def get_stage2_pot_stage(pot_key):
    if pot_key == "left":
        return stage2_left_bush_stage
    if pot_key == "right":
        return stage2_right_bush_stage
    return 0


def get_stage2_pot_position(pot_key):
    if pot_key == "left":
        return STAGE2_LEFT_POT_CENTER_X, STAGE2_LEFT_POT_SOIL_Y
    if pot_key == "right":
        return STAGE2_RIGHT_POT_CENTER_X, STAGE2_RIGHT_POT_SOIL_Y
    return WIDTH // 2, HEIGHT // 2


def get_stage2_character_center(x=None, y=None):
    if x is None:
        x = stage2_sun_target_x
    if y is None:
        y = stage2_sun_target_y
    return int(x + SUN_SIZE / 2), int(y + SUN_SIZE / 2)


def is_stage2_point_on_dirt_road(px, py):
    for rect in STAGE2_DIRT_ROAD_RECTS:
        if point_inside_rect(px, py, rect):
            return True
    return False


def can_stage2_move_to(target_x, target_y):
    """
    Stage 2 free movement gate.
    The sun/cloud is no longer locked to the drawn dirt-road rectangles in Stage 2.
    Movement is allowed anywhere inside the Stage 2 play area; pot trigger rectangles
    still decide when the character has reached a pot.
    """
    return (
        STAGE2_SUN_MIN_X <= target_x <= STAGE2_SUN_MAX_X and
        STAGE2_SUN_MIN_Y <= target_y <= STAGE2_SUN_MAX_Y
    )


def get_stage2_lock_position(pot_key):
    pot_x, pot_y = get_stage2_pot_position(pot_key)
    lock_x = int(pot_x - SUN_SIZE / 2)
    lock_y = int(pot_y - SUN_SIZE - STAGE2_LOCK_GAP_ABOVE_POT)
    lock_x = max(0, min(WIDTH - SUN_SIZE, lock_x))
    lock_y = max(0, min(HEIGHT - SUN_SIZE, lock_y))
    return float(lock_x), float(lock_y)


def reset_stage2_chin_progress():
    global stage2_chin_tuck_total_time
    global stage2_chin_tuck_last_update_time
    stage2_chin_tuck_total_time = 0.0
    stage2_chin_tuck_last_update_time = None


def update_stage2_chin_tuck_progress(is_detected):
    global stage2_chin_tuck_total_time
    global stage2_chin_tuck_last_update_time
    now = time.time()
    if is_detected:
        if stage2_chin_tuck_last_update_time is None:
            stage2_chin_tuck_last_update_time = now
        else:
            elapsed = now - stage2_chin_tuck_last_update_time
            if 0.0 <= elapsed <= 1.0:
                stage2_chin_tuck_total_time += elapsed
            stage2_chin_tuck_last_update_time = now
    else:
        stage2_chin_tuck_last_update_time = None
    if stage2_chin_tuck_total_time > STAGE2_CHIN_REQUIRED_TOTAL_TIME:
        stage2_chin_tuck_total_time = STAGE2_CHIN_REQUIRED_TOTAL_TIME
    return stage2_chin_tuck_total_time


def reset_stage2_shoulder_progress():
    global stage2_shoulder_hold_start
    global stage2_shoulder_release_start_time
    global stage2_shoulder_total_time
    global stage2_shoulder_last_update_time
    stage2_shoulder_hold_start = None
    stage2_shoulder_release_start_time = None
    stage2_shoulder_total_time = 0.0
    stage2_shoulder_last_update_time = None


def pause_stage2_shoulder_progress():
    global stage2_shoulder_hold_start
    global stage2_shoulder_release_start_time
    global stage2_shoulder_last_update_time
    stage2_shoulder_hold_start = None
    stage2_shoulder_release_start_time = None
    stage2_shoulder_last_update_time = None



def reset_stage2_retraction_progress():
    global stage2_retraction_hold_start
    global stage2_retraction_last_seen_time
    global stage2_retraction_total_time
    global stage2_retraction_last_update_time
    stage2_retraction_hold_start = None
    stage2_retraction_last_seen_time = None
    stage2_retraction_total_time = 0.0
    stage2_retraction_last_update_time = None


def pause_stage2_retraction_progress():
    global stage2_retraction_hold_start
    global stage2_retraction_last_seen_time
    global stage2_retraction_last_update_time
    stage2_retraction_hold_start = None
    stage2_retraction_last_seen_time = None
    stage2_retraction_last_update_time = None



def reset_stage2_rain_sequence():
    global stage2_rain_sequence_active
    global stage2_rain_pot_key
    global stage2_rain_start_time
    global stage2_stage3_pause_active
    global stage2_stage3_pause_pot_key
    global stage2_stage3_pause_start_time
    stage2_rain_sequence_active = False
    stage2_rain_pot_key = None
    stage2_rain_start_time = 0.0
    stage2_stage3_pause_active = False
    stage2_stage3_pause_pot_key = None
    stage2_stage3_pause_start_time = 0.0


def set_stage2_pot_stage(pot_key, stage_value):
    global stage2_left_bush_stage
    global stage2_right_bush_stage
    if pot_key == "left":
        stage2_left_bush_stage = int(stage_value)
    elif pot_key == "right":
        stage2_right_bush_stage = int(stage_value)


def all_stage2_pots_fully_grown():
    return stage2_left_bush_stage == 3 and stage2_right_bush_stage == 3


def unlock_stage2_character_to_start():
    global stage2_locked_to_pot
    global stage2_locked_pot_key
    global stage2_active_pot_key
    global stage2_sun_current_x
    global stage2_sun_current_y
    global stage2_sun_target_x
    global stage2_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    stage2_locked_to_pot = False
    stage2_locked_pot_key = None
    stage2_active_pot_key = None
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    stage2_sun_current_x = float(STAGE2_SUN_START_X)
    stage2_sun_current_y = float(STAGE2_SUN_START_Y)
    stage2_sun_target_x = float(STAGE2_SUN_START_X)
    stage2_sun_target_y = float(STAGE2_SUN_START_Y)
    clear_all_movement_holds()
    reset_stage2_chin_progress()
    reset_stage2_shoulder_progress()
    reset_stage2_retraction_progress()
    reset_stage2_rain_sequence()


def lock_stage2_character_above_pot(pot_key):
    global stage2_locked_to_pot
    global stage2_locked_pot_key
    global stage2_active_pot_key
    global stage2_sun_target_x
    global stage2_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    global stage2_message
    lock_x, lock_y = get_stage2_lock_position(pot_key)
    stage2_locked_to_pot = True
    stage2_locked_pot_key = pot_key
    stage2_active_pot_key = pot_key
    stage2_sun_target_x = lock_x
    stage2_sun_target_y = lock_y
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    clear_all_movement_holds()
    reset_stage2_chin_progress()
    reset_stage2_shoulder_progress()
    reset_stage2_retraction_progress()
    reset_stage2_rain_sequence()
    stage2_message = f"Great! {get_stage2_pot_name(pot_key)} started growing. Now hold Chin Tuck for 10 seconds."


def activate_stage2_pot_stage1_and_lock(pot_key):
    stage_before = get_stage2_pot_stage(pot_key)
    if stage_before == 0:
        set_stage2_pot_stage(pot_key, 1)
        lock_stage2_character_above_pot(pot_key)
        return f"{get_stage2_pot_name(pot_key)} Stage 1. Hold Chin Tuck for 10 seconds."
    if stage_before < 3:
        lock_stage2_character_above_pot(pot_key)
        return f"Back to {get_stage2_pot_name(pot_key)}. Continue this pot."
    return f"{get_stage2_pot_name(pot_key)} is complete. Choose the other pot."


def check_stage2_pot_reached():
    cx, cy = get_stage2_character_center()
    if point_inside_rect(cx, cy, STAGE2_LEFT_TRIGGER_RECT):
        return activate_stage2_pot_stage1_and_lock("left")
    if point_inside_rect(cx, cy, STAGE2_RIGHT_TRIGGER_RECT):
        return activate_stage2_pot_stage1_and_lock("right")
    return ""


def reset_stage2_state_keep_calibration():
    global stage2_sun_current_x
    global stage2_sun_current_y
    global stage2_sun_target_x
    global stage2_sun_target_y
    global stage2_left_bush_stage
    global stage2_right_bush_stage
    global stage2_score
    global stage2_locked_to_pot
    global stage2_locked_pot_key
    global stage2_active_pot_key
    global stage2_message
    global stage2_completed
    global stage2_completion_time
    global active_character
    global last_sun_move_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    global cloud_activation_time
    stage2_sun_current_x = float(STAGE2_SUN_START_X)
    stage2_sun_current_y = float(STAGE2_SUN_START_Y)
    stage2_sun_target_x = float(STAGE2_SUN_START_X)
    stage2_sun_target_y = float(STAGE2_SUN_START_Y)
    stage2_left_bush_stage = 0
    stage2_right_bush_stage = 0
    stage2_score = 0
    stage2_locked_to_pot = False
    stage2_locked_pot_key = None
    stage2_active_pot_key = None
    stage2_message = "Move the sun to one of the empty summer pots."
    stage2_completed = False
    stage2_completion_time = None
    active_character = "sun"
    last_sun_move_time = 0
    rain_effect_start_time = 0
    rain_effect_x = float(STAGE2_SUN_START_X)
    rain_effect_y = float(STAGE2_SUN_START_Y)
    cloud_activation_time = 0.0
    clear_all_movement_holds()
    reset_stage2_chin_progress()
    reset_stage2_shoulder_progress()
    reset_stage2_retraction_progress()
    reset_stage2_rain_sequence()


def start_stage2_after_calibration():
    global game_state
    global mouse_left_clicked
    global current_stage_number
    global win_message
    reset_stage2_state_keep_calibration()
    current_stage_number = 2
    win_message = ""
    mouse_left_clicked = False
    start_new_session_metrics("stage_2_play", stage_number=2)
    game_state = "stage2"


def start_stage2_rain_sequence(pot_key):
    global stage2_rain_sequence_active
    global stage2_rain_pot_key
    global stage2_rain_start_time
    global stage2_stage3_pause_active
    global stage2_stage3_pause_pot_key
    global stage2_stage3_pause_start_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    now = time.time()
    stage2_rain_sequence_active = True
    stage2_rain_pot_key = pot_key
    stage2_rain_start_time = now
    stage2_stage3_pause_active = False
    stage2_stage3_pause_pot_key = None
    stage2_stage3_pause_start_time = 0.0
    rain_effect_start_time = now
    rain_effect_x = float(stage2_sun_target_x)
    rain_effect_y = float(stage2_sun_target_y)


def update_stage2_rain_sequence():
    global stage2_rain_sequence_active
    global stage2_rain_pot_key
    global stage2_rain_start_time
    global stage2_stage3_pause_active
    global stage2_stage3_pause_pot_key
    global stage2_stage3_pause_start_time
    global stage2_score
    global stage2_completed
    global stage2_completion_time
    global game_state
    global win_message
    now = time.time()
    if stage2_rain_sequence_active:
        pot_key = stage2_rain_pot_key
        elapsed = now - stage2_rain_start_time
        if elapsed < STAGE2_RAIN_DURATION:
            return f"Rain is falling: {elapsed:.1f}s / {STAGE2_RAIN_DURATION:.1f}s"
        if get_stage2_pot_stage(pot_key) == 2:
            set_stage2_pot_stage(pot_key, 3)
            stage2_score = min(STAGE2_TOTAL_POTS, stage2_score + 1)
        stage2_rain_sequence_active = False
        stage2_rain_pot_key = None
        stage2_rain_start_time = 0.0
        stage2_stage3_pause_active = True
        stage2_stage3_pause_pot_key = pot_key
        stage2_stage3_pause_start_time = now
        return f"{get_stage2_pot_name(pot_key)} is fully grown. Wait a moment."
    if stage2_stage3_pause_active:
        pot_key = stage2_stage3_pause_pot_key
        elapsed = now - stage2_stage3_pause_start_time
        if elapsed < STAGE2_RETURN_DELAY_AFTER_POT:
            remaining = max(0.0, STAGE2_RETURN_DELAY_AFTER_POT - elapsed)
            return f"Beautiful! Returning to the lower path in {remaining:.1f}s."
        reset_stage2_rain_sequence()
        unlock_stage2_character_to_start()
        if all_stage2_pots_fully_grown():
            stage2_completed = True
            stage2_completion_time = now
            return f"Stage 2 complete! Win screen in {STAGE2_WIN_DELAY:.1f}s."
        return "Good job. Move to the other summer pot."
    if stage2_completed and stage2_completion_time is not None:
        elapsed = now - stage2_completion_time
        if elapsed >= STAGE2_WIN_DELAY:
            mark_stage_completed(2)
            finalize_session_save("completed")
            win_message = "Stage 2 complete. Both summer bushes are fully grown."
            game_state = "win"
            return "Stage 2 complete. You won!"
        return f"Stage 2 complete! Win screen in {max(0.0, STAGE2_WIN_DELAY - elapsed):.1f}s."
    return None


def process_stage2_shoulder_to_cloud(current_pitch, current_yaw, current_roll, current_shoulder_features, current_shoulder_meta):
    global stage2_shoulder_total_time
    global stage2_shoulder_last_update_time
    global active_character
    global sun_shining_start_time
    global rain_effect_start_time
    global cloud_activation_time

    required_time = STAGE2_SHOULDER_REQUIRED_HOLD_TIME

    if active_character == "cloud":
        reset_stage2_shoulder_progress()
        return "Cloud is ready. Now move shoulders back for rain."

    if not (
        current_shoulder_features is not None and current_shoulder_meta is not None and
        shoulder_neutral_features is not None and shoulder_target_features is not None and
        shoulder_neutral_width is not None and shoulder_neutral_nose_y is not None and
        shoulder_neutral_angle is not None
    ):
        pause_stage2_shoulder_progress()
        return f"Lift shoulders. Progress saved: {stage2_shoulder_total_time:.1f}s / {required_time:.1f}s"

    head_ready = True
    if current_pitch is not None and neutral_pitch is not None and current_yaw is not None and neutral_yaw is not None and current_roll is not None and neutral_roll is not None:
        head_ready = (
            abs(angle_diff(current_pitch, neutral_pitch)) <= SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_yaw, neutral_yaw)) <= SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_roll, neutral_roll)) <= SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE
        )
    if not head_ready:
        pause_stage2_shoulder_progress()
        return f"Keep head steady. Progress saved: {stage2_shoulder_total_time:.1f}s / {required_time:.1f}s"

    sh_progress, sh_side_error, sh_direct_error, sh_target_strength, sh_current_strength = shoulder_lift_metrics(
        current_shoulder_features,
        shoulder_neutral_features,
        shoulder_target_features
    )
    target_left = shoulder_target_features[0] - shoulder_neutral_features[0]
    target_right = shoulder_target_features[1] - shoulder_neutral_features[1]
    current_left = current_shoulder_features[0] - shoulder_neutral_features[0]
    current_right = current_shoulder_features[1] - shoulder_neutral_features[1]

    continuing = stage2_shoulder_total_time > 0.0 or stage2_shoulder_last_update_time is not None
    ratio = 0.35 if not continuing else 0.20
    weak_ratio = 0.18
    safe_left = max(abs(target_left), SHOULDER_MIN_SINGLE_LIFT)
    safe_right = max(abs(target_right), SHOULDER_MIN_SINGLE_LIFT)
    left_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_left * ratio)
    right_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_right * ratio)
    left_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_left * weak_ratio)
    right_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_right * weak_ratio)

    detected = sh_target_strength >= SHOULDER_MIN_TARGET_STRENGTH and (
        (current_left >= left_req and current_right >= right_req) or
        (current_left >= left_req and current_right >= right_weak) or
        (current_right >= right_req and current_left >= left_weak)
    )

    stage2_shoulder_total_time, stage2_shoulder_last_update_time = update_cumulative_hold_progress(
        detected,
        stage2_shoulder_total_time,
        stage2_shoulder_last_update_time,
        required_time
    )

    if stage2_shoulder_total_time >= required_time:
        active_character = "cloud"
        sun_shining_start_time = 0
        rain_effect_start_time = 0
        cloud_activation_time = time.time()
        clear_chin_histories()
        reset_stage2_shoulder_progress()
        return "Cloud activated! Now move shoulders back for 10 seconds."

    if detected:
        return f"Scapular Elevation total: {stage2_shoulder_total_time:.1f}s / {required_time:.1f}s"

    return (
        f"Scapular Elevation paused; progress saved {stage2_shoulder_total_time:.1f}s/{required_time:.1f}s | "
        f"L {current_left:.3f}/{left_req:.3f} | R {current_right:.3f}/{right_req:.3f}"
    )



def process_stage2_cloud_retraction_rain(current_features, face_detected, palms_detected, shoulders_detected, hands_outside_shoulders, shoulder_gate_info):
    global stage2_retraction_total_time
    global stage2_retraction_last_update_time

    required_time = STAGE2_RETRACTION_REQUIRED_HOLD_TIME

    if not (retraction_neutral_features is not None and retraction_target_features is not None and retraction_calibration_success):
        pause_stage2_retraction_progress()
        return f"Rain movement is not calibrated. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not face_detected:
        pause_stage2_retraction_progress()
        return f"Show your face. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not palms_detected:
        pause_stage2_retraction_progress()
        return f"Show both palms. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not shoulders_detected:
        pause_stage2_retraction_progress()
        return f"Shoulders not detected. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not hands_outside_shoulders:
        pause_stage2_retraction_progress()
        return f"Keep palms outside shoulder width. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"
    if current_features is None:
        pause_stage2_retraction_progress()
        return f"Keep face, shoulders, and palms visible. Progress saved: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"

    averaged = average_retraction_vectors(retraction_current_buffer)
    if averaged is None:
        averaged = current_features.copy()
    detected, score_value, target_strength, current_strength, progress, side_error = is_retraction(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )

    stage2_retraction_total_time, stage2_retraction_last_update_time = update_cumulative_hold_progress(
        detected,
        stage2_retraction_total_time,
        stage2_retraction_last_update_time,
        required_time
    )

    if stage2_retraction_total_time >= required_time:
        start_stage2_rain_sequence(stage2_locked_pot_key)
        reset_stage2_retraction_progress()
        return f"Rain started above {get_stage2_pot_name(stage2_locked_pot_key)}!"

    if detected:
        return f"Scapular Retraction total: {stage2_retraction_total_time:.1f}s / {required_time:.1f}s"

    strict_ok, gate_info, req_gap, req_left, req_right = strict_retraction_gate(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )
    return (
        f"Retraction paused; progress saved {stage2_retraction_total_time:.1f}s/{required_time:.1f}s | "
        f"gap {gate_info['gap_delta']:.2f}/{req_gap:.2f} | "
        f"L {gate_info['left_delta']:.2f}/{req_left:.2f} | R {gate_info['right_delta']:.2f}/{req_right:.2f}"
    )



def process_stage2_free_movement(current_pitch, current_yaw, current_side_bend_angle):
    global stage2_sun_target_x
    global stage2_sun_target_y
    global last_sun_move_time
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    now = time.time()
    if now - last_sun_move_time < STAGE2_MOVE_COOLDOWN:
        return "Move the sun to one of the empty pots."
    # Side bend first, so horizontal movement feels clear on this stage.
    if current_side_bend_angle is not None and neutral_side_bend_angle is not None and left_side_bend_direction is not None and right_side_bend_direction is not None:
        side_bend_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
        left_amount = left_side_bend_direction * side_bend_delta
        right_amount = right_side_bend_direction * side_bend_delta
        left_detected = False
        right_detected = False
        if left_side_bend_threshold is not None and left_amount >= left_side_bend_threshold:
            if left_side_bend_hold_start is None:
                left_side_bend_hold_start = now
            if now - left_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
                left_detected = True
        else:
            left_side_bend_hold_start = None
        if right_side_bend_threshold is not None and right_amount >= right_side_bend_threshold:
            if right_side_bend_hold_start is None:
                right_side_bend_hold_start = now
            if now - right_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME:
                right_detected = True
        else:
            right_side_bend_hold_start = None
        if left_detected or right_detected:
            candidate_x = stage2_sun_target_x + (STAGE2_MOVE_DISTANCE if right_detected else -STAGE2_MOVE_DISTANCE)
            candidate_x = max(STAGE2_SUN_MIN_X, min(STAGE2_SUN_MAX_X, candidate_x))
            if can_stage2_move_to(candidate_x, stage2_sun_target_y):
                stage2_sun_target_x = candidate_x
                last_sun_move_time = now
                left_side_bend_hold_start = None
                right_side_bend_hold_start = None
                pot_msg = check_stage2_pot_reached()
                if pot_msg:
                    return pot_msg
                return "Good side bend. Move freely toward a pot."
            left_side_bend_hold_start = None
            right_side_bend_hold_start = None
            return "Stay inside the Stage 2 play area."
    # Flexion moves forward/down; Extension moves backward/up.
    if current_pitch is not None and neutral_pitch is not None and flexion_direction is not None and extension_direction is not None:
        pitch_delta = angle_diff(current_pitch, neutral_pitch)
        yaw_delta = abs(angle_diff(current_yaw, neutral_yaw)) if current_yaw is not None and neutral_yaw is not None else 0.0
        if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
            flexion_hold_start = None
            extension_hold_start = None
            return "Keep your face forward while moving."
        flexion_amount = flexion_direction * pitch_delta
        extension_amount = extension_direction * pitch_delta
        flexion_detected = False
        extension_detected = False
        if flexion_threshold is not None and flexion_amount >= flexion_threshold:
            if flexion_hold_start is None:
                flexion_hold_start = now
            if now - flexion_hold_start >= FLEXION_REQUIRED_HOLD_TIME:
                flexion_detected = True
        else:
            flexion_hold_start = None
        if extension_threshold is not None and extension_amount >= extension_threshold:
            if extension_hold_start is None:
                extension_hold_start = now
            if now - extension_hold_start >= EXTENSION_REQUIRED_HOLD_TIME:
                extension_detected = True
        else:
            extension_hold_start = None
        if flexion_detected or extension_detected:
            candidate_y = stage2_sun_target_y + (STAGE2_MOVE_DISTANCE if flexion_detected else -STAGE2_MOVE_DISTANCE)
            candidate_y = max(STAGE2_SUN_MIN_Y, min(STAGE2_SUN_MAX_Y, candidate_y))
            if can_stage2_move_to(stage2_sun_target_x, candidate_y):
                stage2_sun_target_y = candidate_y
                last_sun_move_time = now
                flexion_hold_start = None
                extension_hold_start = None
                pot_msg = check_stage2_pot_reached()
                if pot_msg:
                    return pot_msg
                return "Good movement. Move freely toward a pot."
            flexion_hold_start = None
            extension_hold_start = None
            return "Stay inside the Stage 2 play area."
    return "Move the sun to one of the empty pots."


def process_stage2_stage(current_pitch, current_yaw, current_roll, current_side_bend_angle, current_chin_features, current_shoulder_features, current_shoulder_meta, current_retraction_features, retraction_face_detected, retraction_palms_detected, retraction_shoulders_detected, retraction_hands_outside_shoulders, retraction_shoulder_gate_info):
    global stage2_message
    global sun_shining_start_time
    rain_status = update_stage2_rain_sequence()
    if rain_status is not None:
        stage2_message = rain_status
    elif stage2_completed:
        stage2_message = "Stage 2 complete. Great summer work!"
    elif stage2_locked_to_pot:
        pot_stage = get_stage2_pot_stage(stage2_locked_pot_key)
        pot_name = get_stage2_pot_name(stage2_locked_pot_key)
        if pot_stage == 1:
            reset_stage2_shoulder_progress()
            reset_stage2_retraction_progress()
            if current_chin_features is not None and chin_neutral_features is not None and chin_target_features is not None:
                is_chin_tuck, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                    current_chin_features,
                    chin_neutral_features,
                    chin_target_features,
                    current_pitch,
                    neutral_pitch,
                    current_yaw,
                    neutral_yaw,
                    current_roll,
                    neutral_roll
                )
                total = update_stage2_chin_tuck_progress(is_chin_tuck)
                if total >= STAGE2_CHIN_REQUIRED_TOTAL_TIME:
                    set_stage2_pot_stage(stage2_locked_pot_key, 2)
                    reset_stage2_chin_progress()
                    sun_shining_start_time = time.time()
                    stage2_message = f"Chin Tuck complete! {pot_name} is Stage 2. Now lift shoulders for 5 seconds."
                elif is_chin_tuck:
                    stage2_message = f"Hold Chin Tuck: {total:.1f}s / {STAGE2_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
                else:
                    stage2_message = f"Hold Chin Tuck for 10 seconds. Progress: {total:.1f}s / {STAGE2_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
            else:
                stage2_chin_tuck_last_update_time = None
                stage2_message = "Keep your face visible, then hold Chin Tuck."
        elif pot_stage == 2:
            reset_stage2_chin_progress()
            if active_character == "sun":
                stage2_message = process_stage2_shoulder_to_cloud(
                    current_pitch,
                    current_yaw,
                    current_roll,
                    current_shoulder_features,
                    current_shoulder_meta
                )
            else:
                reset_stage2_shoulder_progress()
                stage2_message = process_stage2_cloud_retraction_rain(
                    current_retraction_features,
                    retraction_face_detected,
                    retraction_palms_detected,
                    retraction_shoulders_detected,
                    retraction_hands_outside_shoulders,
                    retraction_shoulder_gate_info
                )
        else:
            stage2_message = "This pot is complete. Move to the other pot."
    else:
        reset_stage2_chin_progress()
        reset_stage2_shoulder_progress()
        reset_stage2_retraction_progress()
        stage2_message = process_stage2_free_movement(current_pitch, current_yaw, current_side_bend_angle)
    return stage2_message


def get_stage2_bush_asset(pot_key, stage):
    if pot_key == "left":
        if stage == 1:
            return stage2_left_bush_stage1_img
        if stage == 2:
            return stage2_left_bush_stage2_img
        if stage >= 3:
            return stage2_left_bush_stage3_img
    if pot_key == "right":
        if stage == 1:
            return stage2_right_bush_stage1_img
        if stage == 2:
            return stage2_right_bush_stage2_img
        if stage >= 3:
            return stage2_right_bush_stage3_img
    return None


def draw_stage2_bush_fallback(frame, pot_key, stage):
    if stage <= 0:
        return frame
    pot_x, pot_y = get_stage2_pot_position(pot_key)
    base_color = (72, 170, 88) if pot_key == "left" else (150, 105, 205)
    flower_color = (80, 160, 255) if pot_key == "left" else (215, 150, 255)
    radius = 28 + stage * 14
    cv2.circle(frame, (int(pot_x), int(pot_y - 48)), radius, base_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(pot_x - radius * 0.55), int(pot_y - 36)), int(radius * 0.72), base_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(pot_x + radius * 0.55), int(pot_y - 36)), int(radius * 0.72), base_color, -1, cv2.LINE_AA)
    if stage >= 2:
        for angle in range(0, 360, 60):
            rad = math.radians(angle)
            fx = int(pot_x + math.cos(rad) * radius * 0.75)
            fy = int(pot_y - 48 + math.sin(rad) * radius * 0.55)
            cv2.circle(frame, (fx, fy), 7, flower_color, -1, cv2.LINE_AA)
    if stage >= 3:
        for angle in range(30, 360, 60):
            rad = math.radians(angle)
            fx = int(pot_x + math.cos(rad) * radius * 0.95)
            fy = int(pot_y - 48 + math.sin(rad) * radius * 0.72)
            cv2.circle(frame, (fx, fy), 8, (255, 230, 90), -1, cv2.LINE_AA)
    return frame


def draw_stage2_bush_on_pot(frame, pot_key, stage):
    if stage <= 0:
        return frame
    asset = get_stage2_bush_asset(pot_key, stage)
    if asset is None:
        return draw_stage2_bush_fallback(frame, pot_key, stage)
    pot_x, pot_y = get_stage2_pot_position(pot_key)
    desired_size = STAGE2_BUSH_SIZE
    if stage == 1:
        desired_size = int(STAGE2_BUSH_SIZE * 0.74)
    elif stage == 2:
        desired_size = int(STAGE2_BUSH_SIZE * 0.90)
    resized = cv2.resize(asset, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
    x = int(pot_x - desired_size / 2)
    y = int(pot_y - desired_size + 18)
    return overlay_transparent(frame, resized, x, y)


def draw_stage2_instruction_card(frame):
    if stage2_completed:
        title = "Stage 2 Complete"
    elif not stage2_locked_to_pot:
        title = "Stage 2 - Summer Pots"
    elif get_stage2_pot_stage(stage2_locked_pot_key) == 1:
        title = "Step 1 - Chin Tuck"
    elif get_stage2_pot_stage(stage2_locked_pot_key) == 2 and active_character == "sun":
        title = "Step 2 - Lift Shoulders"
    elif get_stage2_pot_stage(stage2_locked_pot_key) == 2 and active_character == "cloud":
        title = "Step 3 - Make Rain"
    else:
        title = "Summer Garden"
    x1, y1, x2, y2 = 260, 22, 1020, 105
    draw_transparent_rounded_rect(frame, x1 + 5, y1 + 6, x2 + 5, y2 + 6, (60, 80, 80), alpha=0.20, radius=28)
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 250, 220), alpha=0.88, radius=28)
    draw_rounded_rect(frame, x1, y1, x2, y2, (50, 170, 105), radius=28, thickness=3)
    draw_centered_text(frame, title, WIDTH // 2, y1 + 35, scale=0.72, color=(30, 130, 75), thickness=2)
    lines = wrap_text_lines(stage2_message, x2 - x1 - 80, scale=0.42, thickness=1)
    msg_y = y1 + 64
    for line in lines[:2]:
        draw_centered_text(frame, line, WIDTH // 2, msg_y, scale=0.42, color=(50, 70, 65), thickness=1)
        msg_y += 20


def draw_stage2_progress_card(frame):
    x1, y1, x2, y2 = 28, 24, 232, 104
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 252, 235), alpha=0.88, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (60, 160, 95), radius=24, thickness=3)
    draw_ui_text(frame, "Summer Pots", x1 + 22, y1 + 32, scale=0.50, color=(30, 120, 70), thickness=2)
    draw_ui_text(frame, f"Score: {stage2_score} / {STAGE2_TOTAL_POTS}", x1 + 22, y1 + 62, scale=0.48, color=(50, 70, 80), thickness=2)
    if stage2_left_bush_stage == 3:
        draw_check_icon(frame, x2 - 54, y1 + 54, radius=12)
    if stage2_right_bush_stage == 3:
        draw_check_icon(frame, x2 - 24, y1 + 54, radius=12)


def draw_stage2_screen(current_sun_frame, current_cloud_frame):
    global stage2_sun_current_x
    global stage2_sun_current_y
    if stage2_background_img is not None:
        frame = stage2_background_img.copy()
    else:
        frame = background.copy()
    frame = draw_stage2_bush_on_pot(frame, "left", stage2_left_bush_stage)
    frame = draw_stage2_bush_on_pot(frame, "right", stage2_right_bush_stage)
    rain_is_active = (time.time() - rain_effect_start_time) <= STAGE2_RAIN_DURATION
    if rain_is_active:
        frame = draw_rain(frame, int(rain_effect_x), int(rain_effect_y), SUN_SIZE)
    stage2_sun_current_x += (stage2_sun_target_x - stage2_sun_current_x) * STAGE2_VISUAL_SMOOTHING_FACTOR
    stage2_sun_current_y += (stage2_sun_target_y - stage2_sun_current_y) * STAGE2_VISUAL_SMOOTHING_FACTOR
    sun_is_shining = (
        active_character == "sun" and
        (stage2_chin_tuck_last_update_time is not None or (time.time() - sun_shining_start_time) <= SUN_SHINING_DURATION)
    )
    if active_character == "sun":
        if sun_is_shining:
            frame = draw_sun_glow(frame, int(stage2_sun_current_x), int(stage2_sun_current_y), SUN_SIZE)
        character_frame = current_sun_frame
    else:
        character_frame = current_cloud_frame
    frame = overlay_transparent(frame, character_frame, int(stage2_sun_current_x), int(stage2_sun_current_y))
    draw_stage2_instruction_card(frame)
    draw_stage2_progress_card(frame)
    home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
    frame = draw_home_icon_button(frame, hovered=home_hovered)
    return frame



# -----------------------------
# Stage 3 - Autumn Garden helpers
# -----------------------------
def get_stage3_pot_name(pot_key):
    names = {
        "chrysanthemum": "Autumn Chrysanthemum",
        "maple": "Autumn Maple",
        "purple_bush": "Purple Autumn Bush",
    }
    return names.get(pot_key, "Autumn plant")


def get_stage3_pot_stage(pot_key):
    if pot_key == "chrysanthemum":
        return stage3_chrysanthemum_stage
    if pot_key == "maple":
        return stage3_maple_stage
    if pot_key == "purple_bush":
        return stage3_purple_bush_stage
    return 0


def set_stage3_pot_stage(pot_key, stage_value):
    global stage3_chrysanthemum_stage
    global stage3_maple_stage
    global stage3_purple_bush_stage
    if pot_key == "chrysanthemum":
        stage3_chrysanthemum_stage = int(stage_value)
    elif pot_key == "maple":
        stage3_maple_stage = int(stage_value)
    elif pot_key == "purple_bush":
        stage3_purple_bush_stage = int(stage_value)


def get_stage3_pot_position(pot_key):
    if pot_key == "chrysanthemum":
        return STAGE3_CHRYSANTHEMUM_CENTER_X, STAGE3_CHRYSANTHEMUM_SOIL_Y
    if pot_key == "maple":
        return STAGE3_MAPLE_CENTER_X, STAGE3_MAPLE_SOIL_Y
    if pot_key == "purple_bush":
        return STAGE3_PURPLE_BUSH_CENTER_X, STAGE3_PURPLE_BUSH_SOIL_Y
    return WIDTH // 2, HEIGHT // 2


def get_stage3_character_center(x=None, y=None):
    if x is None:
        x = stage3_sun_target_x
    if y is None:
        y = stage3_sun_target_y
    return int(x + SUN_SIZE / 2), int(y + SUN_SIZE / 2)


def is_stage3_point_on_dirt_road(px, py):
    return any(point_inside_rect(px, py, rect) for rect in STAGE3_DIRT_ROAD_RECTS)


def can_stage3_move_to(target_x, target_y):
    """Allow Stage 3 movement anywhere inside the visible game window."""
    return (
        STAGE3_SUN_MIN_X <= target_x <= STAGE3_SUN_MAX_X and
        STAGE3_SUN_MIN_Y <= target_y <= STAGE3_SUN_MAX_Y
    )


def get_stage3_lock_position(pot_key):
    pot_x, pot_y = get_stage3_pot_position(pot_key)
    gap = STAGE3_MAPLE_LOCK_GAP_ABOVE_POT if pot_key == "maple" else STAGE3_LOCK_GAP_ABOVE_POT
    lock_x = int(pot_x - SUN_SIZE / 2)
    lock_y = int(pot_y - SUN_SIZE - gap)
    lock_x = max(0, min(WIDTH - SUN_SIZE, lock_x))
    lock_y = max(0, min(HEIGHT - SUN_SIZE, lock_y))
    return float(lock_x), float(lock_y)


def reset_stage3_chin_progress():
    global stage3_chin_tuck_total_time
    global stage3_chin_tuck_last_update_time
    stage3_chin_tuck_total_time = 0.0
    stage3_chin_tuck_last_update_time = None


def update_stage3_chin_tuck_progress(is_detected):
    global stage3_chin_tuck_total_time
    global stage3_chin_tuck_last_update_time
    now = time.time()
    if is_detected:
        if stage3_chin_tuck_last_update_time is None:
            stage3_chin_tuck_last_update_time = now
        else:
            elapsed = now - stage3_chin_tuck_last_update_time
            if 0.0 <= elapsed <= 1.0:
                stage3_chin_tuck_total_time += elapsed
            stage3_chin_tuck_last_update_time = now
    else:
        # Cumulative: stopping pauses progress instead of clearing it.
        stage3_chin_tuck_last_update_time = None
    stage3_chin_tuck_total_time = min(
        STAGE3_CHIN_REQUIRED_TOTAL_TIME,
        stage3_chin_tuck_total_time
    )
    return stage3_chin_tuck_total_time


def reset_stage3_shoulder_progress():
    global stage3_shoulder_hold_start
    global stage3_shoulder_release_start_time
    global stage3_shoulder_total_time
    global stage3_shoulder_last_update_time
    stage3_shoulder_hold_start = None
    stage3_shoulder_release_start_time = None
    stage3_shoulder_total_time = 0.0
    stage3_shoulder_last_update_time = None


def pause_stage3_shoulder_progress():
    global stage3_shoulder_hold_start
    global stage3_shoulder_release_start_time
    global stage3_shoulder_last_update_time
    stage3_shoulder_hold_start = None
    stage3_shoulder_release_start_time = None
    stage3_shoulder_last_update_time = None



def reset_stage3_retraction_progress():
    global stage3_retraction_hold_start
    global stage3_retraction_last_seen_time
    global stage3_retraction_total_time
    global stage3_retraction_last_update_time
    stage3_retraction_hold_start = None
    stage3_retraction_last_seen_time = None
    stage3_retraction_total_time = 0.0
    stage3_retraction_last_update_time = None


def pause_stage3_retraction_progress():
    global stage3_retraction_hold_start
    global stage3_retraction_last_seen_time
    global stage3_retraction_last_update_time
    stage3_retraction_hold_start = None
    stage3_retraction_last_seen_time = None
    stage3_retraction_last_update_time = None



def reset_stage3_rain_sequence():
    global stage3_rain_sequence_active
    global stage3_rain_pot_key
    global stage3_rain_start_time
    global stage3_stage3_pause_active
    global stage3_stage3_pause_pot_key
    global stage3_stage3_pause_start_time
    stage3_rain_sequence_active = False
    stage3_rain_pot_key = None
    stage3_rain_start_time = 0.0
    stage3_stage3_pause_active = False
    stage3_stage3_pause_pot_key = None
    stage3_stage3_pause_start_time = 0.0


def all_stage3_pots_fully_grown():
    return (
        stage3_chrysanthemum_stage == 3 and
        stage3_maple_stage == 3 and
        stage3_purple_bush_stage == 3
    )


def unlock_stage3_character_to_entrance():
    global stage3_locked_to_pot
    global stage3_locked_pot_key
    global stage3_active_pot_key
    global stage3_sun_current_x
    global stage3_sun_current_y
    global stage3_sun_target_x
    global stage3_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    stage3_locked_to_pot = False
    stage3_locked_pot_key = None
    stage3_active_pot_key = None
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    # Return instantly to the exact initial point in front of the top entrance.
    stage3_sun_current_x = float(STAGE3_SUN_START_X)
    stage3_sun_current_y = float(STAGE3_SUN_START_Y)
    stage3_sun_target_x = float(STAGE3_SUN_START_X)
    stage3_sun_target_y = float(STAGE3_SUN_START_Y)
    clear_all_movement_holds()
    reset_stage3_chin_progress()
    reset_stage3_shoulder_progress()
    reset_stage3_retraction_progress()
    reset_stage3_rain_sequence()


def lock_stage3_character_above_pot(pot_key):
    global stage3_locked_to_pot
    global stage3_locked_pot_key
    global stage3_active_pot_key
    global stage3_sun_target_x
    global stage3_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    global stage3_message
    lock_x, lock_y = get_stage3_lock_position(pot_key)
    stage3_locked_to_pot = True
    stage3_locked_pot_key = pot_key
    stage3_active_pot_key = pot_key
    stage3_sun_target_x = lock_x
    stage3_sun_target_y = lock_y
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    clear_all_movement_holds()
    reset_stage3_chin_progress()
    reset_stage3_shoulder_progress()
    reset_stage3_retraction_progress()
    reset_stage3_rain_sequence()
    stage3_message = f"{get_stage3_pot_name(pot_key)} is Stage 1. Hold Chin Tuck for 10 seconds."


def activate_stage3_pot_stage1_and_lock(pot_key):
    current_stage = get_stage3_pot_stage(pot_key)
    if current_stage == 0:
        set_stage3_pot_stage(pot_key, 1)
        lock_stage3_character_above_pot(pot_key)
        return f"{get_stage3_pot_name(pot_key)} is now Stage 1. Hold Chin Tuck for 10 seconds."
    if current_stage >= 3:
        return "This autumn plant is already complete. Choose another pot."
    lock_stage3_character_above_pot(pot_key)
    return f"Continue growing {get_stage3_pot_name(pot_key)}."


def check_stage3_pot_reached():
    if active_character != "sun" or stage3_locked_to_pot:
        return ""
    center_x, center_y = get_stage3_character_center()
    candidates = [
        ("maple", STAGE3_MAPLE_TRIGGER_RECT),
        ("chrysanthemum", STAGE3_CHRYSANTHEMUM_TRIGGER_RECT),
        ("purple_bush", STAGE3_PURPLE_BUSH_TRIGGER_RECT),
    ]
    for pot_key, rect in candidates:
        if point_inside_rect(center_x, center_y, rect):
            if get_stage3_pot_stage(pot_key) >= 3:
                return "This plant is complete. Move to another autumn pot."
            return activate_stage3_pot_stage1_and_lock(pot_key)
    return ""


def reset_stage3_state_keep_calibration():
    global stage3_sun_current_x
    global stage3_sun_current_y
    global stage3_sun_target_x
    global stage3_sun_target_y
    global stage3_chrysanthemum_stage
    global stage3_maple_stage
    global stage3_purple_bush_stage
    global stage3_score
    global stage3_locked_to_pot
    global stage3_locked_pot_key
    global stage3_active_pot_key
    global stage3_message
    global stage3_completed
    global stage3_completion_time
    global active_character
    global last_sun_move_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    global cloud_activation_time
    stage3_sun_current_x = float(STAGE3_SUN_START_X)
    stage3_sun_current_y = float(STAGE3_SUN_START_Y)
    stage3_sun_target_x = float(STAGE3_SUN_START_X)
    stage3_sun_target_y = float(STAGE3_SUN_START_Y)
    stage3_chrysanthemum_stage = 0
    stage3_maple_stage = 0
    stage3_purple_bush_stage = 0
    stage3_score = 0
    stage3_locked_to_pot = False
    stage3_locked_pot_key = None
    stage3_active_pot_key = None
    stage3_message = "Move the sun from the entrance to one of the autumn pots."
    stage3_completed = False
    stage3_completion_time = None
    active_character = "sun"
    last_sun_move_time = 0
    rain_effect_start_time = 0
    rain_effect_x = float(STAGE3_SUN_START_X)
    rain_effect_y = float(STAGE3_SUN_START_Y)
    cloud_activation_time = 0.0
    clear_all_movement_holds()
    reset_stage3_chin_progress()
    reset_stage3_shoulder_progress()
    reset_stage3_retraction_progress()
    reset_stage3_rain_sequence()


def start_stage3_after_calibration():
    global game_state
    global mouse_left_clicked
    global current_stage_number
    global win_message
    reset_stage3_state_keep_calibration()
    current_stage_number = 3
    win_message = ""
    mouse_left_clicked = False
    start_new_session_metrics("stage_3_play", stage_number=3)
    game_state = "stage3"


def start_stage3_rain_sequence(pot_key):
    global stage3_rain_sequence_active
    global stage3_rain_pot_key
    global stage3_rain_start_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    now = time.time()
    stage3_rain_sequence_active = True
    stage3_rain_pot_key = pot_key
    stage3_rain_start_time = now
    rain_effect_start_time = now
    rain_effect_x = float(stage3_sun_target_x)
    rain_effect_y = float(stage3_sun_target_y)


def update_stage3_rain_sequence():
    global stage3_rain_sequence_active
    global stage3_rain_pot_key
    global stage3_rain_start_time
    global stage3_stage3_pause_active
    global stage3_stage3_pause_pot_key
    global stage3_stage3_pause_start_time
    global stage3_score
    global stage3_completed
    global stage3_completion_time
    global game_state
    global win_message
    now = time.time()
    if stage3_rain_sequence_active:
        pot_key = stage3_rain_pot_key
        elapsed = now - stage3_rain_start_time
        if elapsed < STAGE3_RAIN_DURATION:
            return f"Rain is falling: {elapsed:.1f}s / {STAGE3_RAIN_DURATION:.1f}s"
        if get_stage3_pot_stage(pot_key) == 2:
            set_stage3_pot_stage(pot_key, 3)
            stage3_score = min(STAGE3_TOTAL_POTS, stage3_score + 1)
        stage3_rain_sequence_active = False
        stage3_rain_pot_key = None
        stage3_rain_start_time = 0.0
        stage3_stage3_pause_active = True
        stage3_stage3_pause_pot_key = pot_key
        stage3_stage3_pause_start_time = now
        return f"{get_stage3_pot_name(pot_key)} is fully grown. Wait a moment."
    if stage3_stage3_pause_active:
        elapsed = now - stage3_stage3_pause_start_time
        if elapsed < STAGE3_RETURN_DELAY_AFTER_POT:
            remaining = max(0.0, STAGE3_RETURN_DELAY_AFTER_POT - elapsed)
            return f"Beautiful! Returning to the entrance in {remaining:.1f}s."
        reset_stage3_rain_sequence()
        unlock_stage3_character_to_entrance()
        if all_stage3_pots_fully_grown():
            stage3_completed = True
            stage3_completion_time = now
            return f"Stage 3 complete! Win screen in {STAGE3_WIN_DELAY:.1f}s."
        return "Good job. The sun returned to the entrance. Choose another pot."
    if stage3_completed and stage3_completion_time is not None:
        elapsed = now - stage3_completion_time
        if elapsed >= STAGE3_WIN_DELAY:
            mark_stage_completed(3)
            finalize_session_save("completed")
            win_message = "Stage 3 complete. All autumn plants are fully grown."
            game_state = "win"
            return "Stage 3 complete. You won!"
        return f"Stage 3 complete! Win screen in {max(0.0, STAGE3_WIN_DELAY - elapsed):.1f}s."
    return None


def process_stage3_shoulder_to_cloud(current_pitch, current_yaw, current_roll, current_shoulder_features, current_shoulder_meta):
    global stage3_shoulder_total_time
    global stage3_shoulder_last_update_time
    global active_character
    global sun_shining_start_time
    global rain_effect_start_time
    global cloud_activation_time

    required_time = STAGE3_SHOULDER_REQUIRED_HOLD_TIME

    if active_character == "cloud":
        reset_stage3_shoulder_progress()
        return "Cloud is ready. Now move shoulders back for rain."

    if not (
        current_shoulder_features is not None and current_shoulder_meta is not None and
        shoulder_neutral_features is not None and shoulder_target_features is not None and
        shoulder_neutral_width is not None and shoulder_neutral_nose_y is not None and
        shoulder_neutral_angle is not None
    ):
        pause_stage3_shoulder_progress()
        return f"Lift shoulders. Progress saved: {stage3_shoulder_total_time:.1f}s / {required_time:.1f}s"

    head_ready = True
    if current_pitch is not None and neutral_pitch is not None and current_yaw is not None and neutral_yaw is not None and current_roll is not None and neutral_roll is not None:
        head_ready = (
            abs(angle_diff(current_pitch, neutral_pitch)) <= SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_yaw, neutral_yaw)) <= SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_roll, neutral_roll)) <= SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE
        )
    if not head_ready:
        pause_stage3_shoulder_progress()
        return f"Keep head steady. Progress saved: {stage3_shoulder_total_time:.1f}s / {required_time:.1f}s"

    sh_progress, sh_side_error, sh_direct_error, sh_target_strength, sh_current_strength = shoulder_lift_metrics(
        current_shoulder_features,
        shoulder_neutral_features,
        shoulder_target_features
    )
    target_left = shoulder_target_features[0] - shoulder_neutral_features[0]
    target_right = shoulder_target_features[1] - shoulder_neutral_features[1]
    current_left = current_shoulder_features[0] - shoulder_neutral_features[0]
    current_right = current_shoulder_features[1] - shoulder_neutral_features[1]

    continuing = stage3_shoulder_total_time > 0.0 or stage3_shoulder_last_update_time is not None
    ratio = 0.35 if not continuing else 0.20
    weak_ratio = 0.18
    safe_left = max(abs(target_left), SHOULDER_MIN_SINGLE_LIFT)
    safe_right = max(abs(target_right), SHOULDER_MIN_SINGLE_LIFT)
    left_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_left * ratio)
    right_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_right * ratio)
    left_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_left * weak_ratio)
    right_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_right * weak_ratio)

    detected = sh_target_strength >= SHOULDER_MIN_TARGET_STRENGTH and (
        (current_left >= left_req and current_right >= right_req) or
        (current_left >= left_req and current_right >= right_weak) or
        (current_right >= right_req and current_left >= left_weak)
    )

    stage3_shoulder_total_time, stage3_shoulder_last_update_time = update_cumulative_hold_progress(
        detected,
        stage3_shoulder_total_time,
        stage3_shoulder_last_update_time,
        required_time
    )

    if stage3_shoulder_total_time >= required_time:
        active_character = "cloud"
        sun_shining_start_time = 0
        rain_effect_start_time = 0
        cloud_activation_time = time.time()
        clear_chin_histories()
        reset_stage3_shoulder_progress()
        return "Cloud activated! Now move shoulders back for 10 seconds."

    if detected:
        return f"Scapular Elevation total: {stage3_shoulder_total_time:.1f}s / {required_time:.1f}s"

    return (
        f"Scapular Elevation paused; progress saved {stage3_shoulder_total_time:.1f}s/{required_time:.1f}s | "
        f"L {current_left:.3f}/{left_req:.3f} | R {current_right:.3f}/{right_req:.3f}"
    )



def process_stage3_cloud_retraction_rain(current_features, face_detected, palms_detected, shoulders_detected, hands_outside_shoulders, shoulder_gate_info):
    global stage3_retraction_total_time
    global stage3_retraction_last_update_time

    required_time = STAGE3_RETRACTION_REQUIRED_HOLD_TIME

    if not (retraction_neutral_features is not None and retraction_target_features is not None and retraction_calibration_success):
        pause_stage3_retraction_progress()
        return f"Rain movement is not calibrated. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not face_detected:
        pause_stage3_retraction_progress()
        return f"Show your face. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not palms_detected:
        pause_stage3_retraction_progress()
        return f"Show both palms. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not shoulders_detected:
        pause_stage3_retraction_progress()
        return f"Shoulders not detected. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not hands_outside_shoulders:
        pause_stage3_retraction_progress()
        return f"Keep palms outside shoulder width. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"
    if current_features is None:
        pause_stage3_retraction_progress()
        return f"Keep face, shoulders, and palms visible. Progress saved: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"

    averaged = average_retraction_vectors(retraction_current_buffer)
    if averaged is None:
        averaged = current_features.copy()
    detected, score_value, target_strength, current_strength, progress, side_error = is_retraction(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )

    stage3_retraction_total_time, stage3_retraction_last_update_time = update_cumulative_hold_progress(
        detected,
        stage3_retraction_total_time,
        stage3_retraction_last_update_time,
        required_time
    )

    if stage3_retraction_total_time >= required_time:
        start_stage3_rain_sequence(stage3_locked_pot_key)
        reset_stage3_retraction_progress()
        return f"Rain started above {get_stage3_pot_name(stage3_locked_pot_key)}!"

    if detected:
        return f"Scapular Retraction total: {stage3_retraction_total_time:.1f}s / {required_time:.1f}s"

    strict_ok, gate_info, req_gap, req_left, req_right = strict_retraction_gate(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )
    return (
        f"Retraction paused; progress saved {stage3_retraction_total_time:.1f}s/{required_time:.1f}s | "
        f"gap {gate_info['gap_delta']:.2f}/{req_gap:.2f} | "
        f"L {gate_info['left_delta']:.2f}/{req_left:.2f} | R {gate_info['right_delta']:.2f}/{req_right:.2f}"
    )



def _apply_stage3_move(candidate_x, candidate_y):
    global stage3_sun_target_x
    global stage3_sun_target_y
    global last_sun_move_time
    if not can_stage3_move_to(candidate_x, candidate_y):
        return False, "Cannot move outside the game area."
    stage3_sun_target_x = float(candidate_x)
    stage3_sun_target_y = float(candidate_y)
    last_sun_move_time = time.time()
    pot_message = check_stage3_pot_reached()
    return True, pot_message if pot_message else "Good movement. Move freely toward an autumn pot."


def process_stage3_free_movement(current_pitch, current_yaw, current_side_bend_angle):
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    now = time.time()
    if now - last_sun_move_time < STAGE3_MOVE_COOLDOWN:
        return "Move the sun from the entrance to an unfinished autumn pot."

    # Horizontal movement: left/right side bend.
    if current_side_bend_angle is not None and neutral_side_bend_angle is not None and left_side_bend_direction is not None and right_side_bend_direction is not None:
        delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
        left_amount = left_side_bend_direction * delta
        right_amount = right_side_bend_direction * delta
        left_detected = False
        right_detected = False
        if left_side_bend_threshold is not None and left_amount >= left_side_bend_threshold:
            if left_side_bend_hold_start is None:
                left_side_bend_hold_start = now
            left_detected = now - left_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME
        else:
            left_side_bend_hold_start = None
        if right_side_bend_threshold is not None and right_amount >= right_side_bend_threshold:
            if right_side_bend_hold_start is None:
                right_side_bend_hold_start = now
            right_detected = now - right_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME
        else:
            right_side_bend_hold_start = None
        if left_detected or right_detected:
            direction = 1 if right_detected else -1
            candidate_x = stage3_sun_target_x + direction * STAGE3_MOVE_DISTANCE
            candidate_x = max(STAGE3_SUN_MIN_X, min(STAGE3_SUN_MAX_X, candidate_x))
            moved, message = _apply_stage3_move(candidate_x, stage3_sun_target_y)
            left_side_bend_hold_start = None
            right_side_bend_hold_start = None
            return message

    # Vertical movement: Flexion moves forward/down; Extension moves backward/up.
    if current_pitch is not None and neutral_pitch is not None and flexion_direction is not None and extension_direction is not None:
        pitch_delta = angle_diff(current_pitch, neutral_pitch)
        yaw_delta = abs(angle_diff(current_yaw, neutral_yaw)) if current_yaw is not None and neutral_yaw is not None else 0.0
        if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
            flexion_hold_start = None
            extension_hold_start = None
            return "Keep your face forward while moving."
        flexion_amount = flexion_direction * pitch_delta
        extension_amount = extension_direction * pitch_delta
        flexion_detected = False
        extension_detected = False
        if flexion_threshold is not None and flexion_amount >= flexion_threshold:
            if flexion_hold_start is None:
                flexion_hold_start = now
            flexion_detected = now - flexion_hold_start >= FLEXION_REQUIRED_HOLD_TIME
        else:
            flexion_hold_start = None
        if extension_threshold is not None and extension_amount >= extension_threshold:
            if extension_hold_start is None:
                extension_hold_start = now
            extension_detected = now - extension_hold_start >= EXTENSION_REQUIRED_HOLD_TIME
        else:
            extension_hold_start = None
        if flexion_detected or extension_detected:
            direction = 1 if flexion_detected else -1
            candidate_y = stage3_sun_target_y + direction * STAGE3_MOVE_DISTANCE
            candidate_y = max(STAGE3_SUN_MIN_Y, min(STAGE3_SUN_MAX_Y, candidate_y))
            moved, message = _apply_stage3_move(stage3_sun_target_x, candidate_y)
            flexion_hold_start = None
            extension_hold_start = None
            return message

    return "Move the sun from the entrance to an unfinished autumn pot."


def process_stage3_stage(current_pitch, current_yaw, current_roll, current_side_bend_angle, current_chin_features, current_shoulder_features, current_shoulder_meta, current_retraction_features, retraction_face_detected, retraction_palms_detected, retraction_shoulders_detected, retraction_hands_outside_shoulders, retraction_shoulder_gate_info):
    global stage3_message
    global sun_shining_start_time
    rain_status = update_stage3_rain_sequence()
    if rain_status is not None:
        stage3_message = rain_status
    elif stage3_completed:
        stage3_message = "Stage 3 complete. Great autumn work!"
    elif stage3_locked_to_pot:
        pot_stage = get_stage3_pot_stage(stage3_locked_pot_key)
        pot_name = get_stage3_pot_name(stage3_locked_pot_key)
        if pot_stage == 1:
            reset_stage3_shoulder_progress()
            reset_stage3_retraction_progress()
            if current_chin_features is not None and chin_neutral_features is not None and chin_target_features is not None:
                is_chin, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                    current_chin_features,
                    chin_neutral_features,
                    chin_target_features,
                    current_pitch,
                    neutral_pitch,
                    current_yaw,
                    neutral_yaw,
                    current_roll,
                    neutral_roll
                )
                total = update_stage3_chin_tuck_progress(is_chin)
                if total >= STAGE3_CHIN_REQUIRED_TOTAL_TIME:
                    set_stage3_pot_stage(stage3_locked_pot_key, 2)
                    reset_stage3_chin_progress()
                    sun_shining_start_time = time.time()
                    stage3_message = f"Chin Tuck complete! {pot_name} is Stage 2. Lift shoulders for 5 seconds."
                elif is_chin:
                    stage3_message = f"Hold Chin Tuck: {total:.1f}s / {STAGE3_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
                else:
                    stage3_message = f"Hold Chin Tuck for 10 seconds. Progress: {total:.1f}s / {STAGE3_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
            else:
                update_stage3_chin_tuck_progress(False)
                stage3_message = "Keep your face visible, then hold Chin Tuck."
        elif pot_stage == 2:
            reset_stage3_chin_progress()
            if active_character == "sun":
                stage3_message = process_stage3_shoulder_to_cloud(
                    current_pitch,
                    current_yaw,
                    current_roll,
                    current_shoulder_features,
                    current_shoulder_meta
                )
            else:
                reset_stage3_shoulder_progress()
                stage3_message = process_stage3_cloud_retraction_rain(
                    current_retraction_features,
                    retraction_face_detected,
                    retraction_palms_detected,
                    retraction_shoulders_detected,
                    retraction_hands_outside_shoulders,
                    retraction_shoulder_gate_info
                )
        else:
            stage3_message = "This plant is complete. Returning to the entrance."
    else:
        reset_stage3_chin_progress()
        reset_stage3_shoulder_progress()
        reset_stage3_retraction_progress()
        stage3_message = process_stage3_free_movement(current_pitch, current_yaw, current_side_bend_angle)
    return stage3_message


def get_stage3_plant_asset(pot_key, stage):
    groups = {
        "chrysanthemum": (
            stage3_chrysanthemum_stage1_img,
            stage3_chrysanthemum_stage2_img,
            stage3_chrysanthemum_stage3_img,
        ),
        "maple": (
            stage3_maple_stage1_img,
            stage3_maple_stage2_img,
            stage3_maple_stage3_img,
        ),
        "purple_bush": (
            stage3_purple_bush_stage1_img,
            stage3_purple_bush_stage2_img,
            stage3_purple_bush_stage3_img,
        ),
    }
    assets = groups.get(pot_key)
    if assets is None or stage <= 0:
        return None
    return assets[min(2, stage - 1)]


def draw_stage3_plant_fallback(frame, pot_key, stage):
    if stage <= 0:
        return frame
    pot_x, pot_y = get_stage3_pot_position(pot_key)
    colors = {
        "chrysanthemum": ((70, 150, 80), (40, 170, 245)),
        "maple": ((65, 125, 75), (35, 105, 235)),
        "purple_bush": ((90, 145, 80), (190, 90, 210)),
    }
    leaf_color, flower_color = colors.get(pot_key, ((70, 150, 80), (80, 180, 240)))
    radius = 25 + stage * 14
    cv2.circle(frame, (int(pot_x), int(pot_y - 45)), radius, leaf_color, -1, cv2.LINE_AA)
    if pot_key == "maple":
        cv2.rectangle(frame, (int(pot_x - 7), int(pot_y - 100 - stage * 15)), (int(pot_x + 7), int(pot_y)), (70, 75, 105), -1)
    for angle in range(0, 360, max(35, 80 - stage * 15)):
        rad = math.radians(angle)
        fx = int(pot_x + math.cos(rad) * radius * 0.75)
        fy = int(pot_y - 48 + math.sin(rad) * radius * 0.58)
        cv2.circle(frame, (fx, fy), 6 + stage, flower_color, -1, cv2.LINE_AA)
    return frame


def draw_stage3_plant_on_pot(frame, pot_key, stage):
    if stage <= 0:
        return frame
    asset = get_stage3_plant_asset(pot_key, stage)
    if asset is None:
        return draw_stage3_plant_fallback(frame, pot_key, stage)
    pot_x, pot_y = get_stage3_pot_position(pot_key)
    base_size = STAGE3_MAPLE_SIZE if pot_key == "maple" else STAGE3_PLANT_SIZE
    factor = 0.70 if stage == 1 else (0.87 if stage == 2 else 1.0)
    desired_size = max(20, int(base_size * factor))
    resized = cv2.resize(asset, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
    x = int(pot_x - desired_size / 2)
    y = int(pot_y - desired_size + 18)
    return overlay_transparent(frame, resized, x, y)


def draw_stage3_scene_base():
    if stage3_background_img is not None:
        frame = stage3_background_img.copy()
    else:
        frame = background.copy()
    frame = draw_stage3_plant_on_pot(frame, "chrysanthemum", stage3_chrysanthemum_stage)
    frame = draw_stage3_plant_on_pot(frame, "maple", stage3_maple_stage)
    frame = draw_stage3_plant_on_pot(frame, "purple_bush", stage3_purple_bush_stage)
    return frame


def draw_stage3_instruction_card(frame):
    if stage3_completed:
        title = "Stage 3 Complete"
    elif not stage3_locked_to_pot:
        title = "Stage 3 - Autumn Garden"
    elif get_stage3_pot_stage(stage3_locked_pot_key) == 1:
        title = "Step 1 - Chin Tuck"
    elif get_stage3_pot_stage(stage3_locked_pot_key) == 2 and active_character == "sun":
        title = "Step 2 - Lift Shoulders"
    else:
        title = "Step 3 - Make Rain"
    # The upper maple and its cloud occupy the top-right area, so its instruction
    # card moves to the bottom to keep the character visible.
    if stage3_locked_to_pot and stage3_locked_pot_key == "maple":
        x1, y1, x2, y2 = 255, 612, 1025, 698
    else:
        x1, y1, x2, y2 = 255, 20, 1025, 106
    draw_transparent_rounded_rect(frame, x1 + 5, y1 + 6, x2 + 5, y2 + 6, (70, 65, 55), alpha=0.20, radius=28)
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 244, 214), alpha=0.90, radius=28)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 125, 190), radius=28, thickness=3)
    draw_centered_text(frame, title, WIDTH // 2, y1 + 35, scale=0.70, color=(40, 95, 155), thickness=2)
    lines = wrap_text_lines(stage3_message, x2 - x1 - 80, scale=0.40, thickness=1)
    msg_y = y1 + 64
    for line in lines[:2]:
        draw_centered_text(frame, line, WIDTH // 2, msg_y, scale=0.40, color=(55, 60, 60), thickness=1)
        msg_y += 19


def draw_stage3_progress_card(frame):
    x1, y1, x2, y2 = 24, 22, 236, 108
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 247, 220), alpha=0.90, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (65, 125, 190), radius=24, thickness=3)
    draw_ui_text(frame, "Autumn Garden", x1 + 20, y1 + 33, scale=0.49, color=(45, 90, 145), thickness=2)
    draw_ui_text(frame, f"Score: {stage3_score} / {STAGE3_TOTAL_POTS}", x1 + 20, y1 + 65, scale=0.47, color=(55, 65, 75), thickness=2)
    completed = [stage3_chrysanthemum_stage, stage3_maple_stage, stage3_purple_bush_stage]
    for i, plant_stage in enumerate(completed):
        if plant_stage == 3:
            draw_check_icon(frame, x2 - 76 + i * 26, y1 + 62, radius=10)


def draw_stage3_screen(current_sun_frame, current_cloud_frame):
    global stage3_sun_current_x
    global stage3_sun_current_y
    frame = draw_stage3_scene_base()
    if (time.time() - rain_effect_start_time) <= STAGE3_RAIN_DURATION:
        frame = draw_rain(frame, int(rain_effect_x), int(rain_effect_y), SUN_SIZE)
    stage3_sun_current_x += (stage3_sun_target_x - stage3_sun_current_x) * STAGE3_VISUAL_SMOOTHING_FACTOR
    stage3_sun_current_y += (stage3_sun_target_y - stage3_sun_current_y) * STAGE3_VISUAL_SMOOTHING_FACTOR
    sun_is_shining = (
        active_character == "sun" and
        (stage3_chin_tuck_last_update_time is not None or (time.time() - sun_shining_start_time) <= SUN_SHINING_DURATION)
    )
    if active_character == "sun":
        if sun_is_shining:
            frame = draw_sun_glow(frame, int(stage3_sun_current_x), int(stage3_sun_current_y), SUN_SIZE)
        character_frame = current_sun_frame
    else:
        character_frame = current_cloud_frame
    frame = overlay_transparent(frame, character_frame, int(stage3_sun_current_x), int(stage3_sun_current_y))
    draw_stage3_instruction_card(frame)
    draw_stage3_progress_card(frame)
    home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
    return draw_home_icon_button(frame, hovered=home_hovered)


# -----------------------------
# Stage 4 - Winter Garden helpers
# -----------------------------
def get_stage4_pot_name(pot_key):
    names = {
        "winter_rose": "Winter Rose",
        "snowdrop": "Snowdrop",
        "poinsettia": "Poinsettia",
        "cyclamen": "Cyclamen",
    }
    return names.get(pot_key, "Winter flower")


def get_stage4_pot_stage(pot_key):
    if pot_key == "winter_rose":
        return stage4_winter_rose_stage
    if pot_key == "snowdrop":
        return stage4_snowdrop_stage
    if pot_key == "poinsettia":
        return stage4_poinsettia_stage
    if pot_key == "cyclamen":
        return stage4_cyclamen_stage
    return 0


def set_stage4_pot_stage(pot_key, stage_value):
    global stage4_winter_rose_stage
    global stage4_snowdrop_stage
    global stage4_poinsettia_stage
    global stage4_cyclamen_stage
    value = int(stage_value)
    if pot_key == "winter_rose":
        stage4_winter_rose_stage = value
    elif pot_key == "snowdrop":
        stage4_snowdrop_stage = value
    elif pot_key == "poinsettia":
        stage4_poinsettia_stage = value
    elif pot_key == "cyclamen":
        stage4_cyclamen_stage = value


def get_stage4_pot_position(pot_key):
    if pot_key == "winter_rose":
        return STAGE4_WINTER_ROSE_CENTER_X, STAGE4_WINTER_ROSE_SOIL_Y
    if pot_key == "snowdrop":
        return STAGE4_SNOWDROP_CENTER_X, STAGE4_SNOWDROP_SOIL_Y
    if pot_key == "poinsettia":
        return STAGE4_POINSETTIA_CENTER_X, STAGE4_POINSETTIA_SOIL_Y
    if pot_key == "cyclamen":
        return STAGE4_CYCLAMEN_CENTER_X, STAGE4_CYCLAMEN_SOIL_Y
    return WIDTH // 2, HEIGHT // 2


def get_stage4_character_center(x=None, y=None):
    if x is None:
        x = stage4_sun_target_x
    if y is None:
        y = stage4_sun_target_y
    return int(x + SUN_SIZE / 2), int(y + SUN_SIZE / 2)


def is_stage4_point_on_dirt_road(px, py):
    return any(point_inside_rect(px, py, rect) for rect in STAGE4_DIRT_ROAD_RECTS)


def can_stage4_move_to(target_x, target_y):
    """Allow Stage 4 movement anywhere inside the visible game window."""
    return (
        STAGE4_SUN_MIN_X <= target_x <= STAGE4_SUN_MAX_X and
        STAGE4_SUN_MIN_Y <= target_y <= STAGE4_SUN_MAX_Y
    )


def get_stage4_lock_position(pot_key):
    pot_x, pot_y = get_stage4_pot_position(pot_key)
    gap = STAGE4_SNOWDROP_LOCK_GAP_ABOVE_POT if pot_key == "snowdrop" else STAGE4_LOCK_GAP_ABOVE_POT
    lock_x = int(pot_x - SUN_SIZE / 2)
    lock_y = int(pot_y - SUN_SIZE - gap)
    lock_x = max(0, min(WIDTH - SUN_SIZE, lock_x))
    lock_y = max(0, min(HEIGHT - SUN_SIZE, lock_y))
    return float(lock_x), float(lock_y)


def reset_stage4_chin_progress():
    global stage4_chin_tuck_total_time
    global stage4_chin_tuck_last_update_time
    stage4_chin_tuck_total_time = 0.0
    stage4_chin_tuck_last_update_time = None


def update_stage4_chin_tuck_progress(is_detected):
    global stage4_chin_tuck_total_time
    global stage4_chin_tuck_last_update_time
    now = time.time()
    if is_detected:
        if stage4_chin_tuck_last_update_time is None:
            stage4_chin_tuck_last_update_time = now
        else:
            elapsed = now - stage4_chin_tuck_last_update_time
            if 0.0 <= elapsed <= 1.0:
                stage4_chin_tuck_total_time += elapsed
            stage4_chin_tuck_last_update_time = now
    else:
        # Cumulative timer: stopping pauses progress instead of clearing it.
        stage4_chin_tuck_last_update_time = None
    stage4_chin_tuck_total_time = min(
        STAGE4_CHIN_REQUIRED_TOTAL_TIME,
        stage4_chin_tuck_total_time
    )
    return stage4_chin_tuck_total_time


def reset_stage4_shoulder_progress():
    global stage4_shoulder_hold_start
    global stage4_shoulder_release_start_time
    global stage4_shoulder_total_time
    global stage4_shoulder_last_update_time
    stage4_shoulder_hold_start = None
    stage4_shoulder_release_start_time = None
    stage4_shoulder_total_time = 0.0
    stage4_shoulder_last_update_time = None


def pause_stage4_shoulder_progress():
    global stage4_shoulder_hold_start
    global stage4_shoulder_release_start_time
    global stage4_shoulder_last_update_time
    stage4_shoulder_hold_start = None
    stage4_shoulder_release_start_time = None
    stage4_shoulder_last_update_time = None



def reset_stage4_retraction_progress():
    global stage4_retraction_hold_start
    global stage4_retraction_last_seen_time
    global stage4_retraction_total_time
    global stage4_retraction_last_update_time
    stage4_retraction_hold_start = None
    stage4_retraction_last_seen_time = None
    stage4_retraction_total_time = 0.0
    stage4_retraction_last_update_time = None


def pause_stage4_retraction_progress():
    global stage4_retraction_hold_start
    global stage4_retraction_last_seen_time
    global stage4_retraction_last_update_time
    stage4_retraction_hold_start = None
    stage4_retraction_last_seen_time = None
    stage4_retraction_last_update_time = None



def reset_stage4_rain_sequence():
    global stage4_rain_sequence_active
    global stage4_rain_pot_key
    global stage4_rain_start_time
    global stage4_stage3_pause_active
    global stage4_stage3_pause_pot_key
    global stage4_stage3_pause_start_time
    stage4_rain_sequence_active = False
    stage4_rain_pot_key = None
    stage4_rain_start_time = 0.0
    stage4_stage3_pause_active = False
    stage4_stage3_pause_pot_key = None
    stage4_stage3_pause_start_time = 0.0


def all_stage4_pots_fully_grown():
    return (
        stage4_winter_rose_stage == 3 and
        stage4_snowdrop_stage == 3 and
        stage4_poinsettia_stage == 3 and
        stage4_cyclamen_stage == 3
    )


def unlock_stage4_character_to_entrance():
    global stage4_locked_to_pot
    global stage4_locked_pot_key
    global stage4_active_pot_key
    global stage4_sun_current_x
    global stage4_sun_current_y
    global stage4_sun_target_x
    global stage4_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    stage4_locked_to_pot = False
    stage4_locked_pot_key = None
    stage4_active_pot_key = None
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    stage4_sun_current_x = float(STAGE4_SUN_START_X)
    stage4_sun_current_y = float(STAGE4_SUN_START_Y)
    stage4_sun_target_x = float(STAGE4_SUN_START_X)
    stage4_sun_target_y = float(STAGE4_SUN_START_Y)
    clear_all_movement_holds()
    reset_stage4_chin_progress()
    reset_stage4_shoulder_progress()
    reset_stage4_retraction_progress()
    reset_stage4_rain_sequence()


def lock_stage4_character_above_pot(pot_key):
    global stage4_locked_to_pot
    global stage4_locked_pot_key
    global stage4_active_pot_key
    global stage4_sun_target_x
    global stage4_sun_target_y
    global active_character
    global cloud_activation_time
    global sun_shining_start_time
    global rain_effect_start_time
    global stage4_message
    lock_x, lock_y = get_stage4_lock_position(pot_key)
    stage4_locked_to_pot = True
    stage4_locked_pot_key = pot_key
    stage4_active_pot_key = pot_key
    stage4_sun_target_x = lock_x
    stage4_sun_target_y = lock_y
    active_character = "sun"
    cloud_activation_time = 0.0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    clear_all_movement_holds()
    reset_stage4_chin_progress()
    reset_stage4_shoulder_progress()
    reset_stage4_retraction_progress()
    reset_stage4_rain_sequence()
    stage4_message = (
        f"{get_stage4_pot_name(pot_key)} is Stage 1 and still frozen. "
        "Hold Chin Tuck for 10 seconds."
    )


def activate_stage4_pot_stage1_and_lock(pot_key):
    current_stage = get_stage4_pot_stage(pot_key)
    if current_stage == 0:
        set_stage4_pot_stage(pot_key, 1)
        lock_stage4_character_above_pot(pot_key)
        return (
            f"{get_stage4_pot_name(pot_key)} is now Stage 1 and frozen. "
            "Hold Chin Tuck for 10 seconds."
        )
    if current_stage >= 3:
        return "This winter flower is already complete. Choose another pot."
    lock_stage4_character_above_pot(pot_key)
    return f"Continue growing {get_stage4_pot_name(pot_key)}."


def check_stage4_pot_reached():
    if active_character != "sun" or stage4_locked_to_pot:
        return ""
    center_x, center_y = get_stage4_character_center()
    candidates = [
        ("snowdrop", STAGE4_SNOWDROP_TRIGGER_RECT),
        ("winter_rose", STAGE4_WINTER_ROSE_TRIGGER_RECT),
        ("poinsettia", STAGE4_POINSETTIA_TRIGGER_RECT),
        ("cyclamen", STAGE4_CYCLAMEN_TRIGGER_RECT),
    ]
    for pot_key, rect in candidates:
        if point_inside_rect(center_x, center_y, rect):
            if get_stage4_pot_stage(pot_key) >= 3:
                return "This flower is complete. Move to another winter pot."
            return activate_stage4_pot_stage1_and_lock(pot_key)
    return ""


def reset_stage4_state_keep_calibration():
    global stage4_sun_current_x
    global stage4_sun_current_y
    global stage4_sun_target_x
    global stage4_sun_target_y
    global stage4_winter_rose_stage
    global stage4_snowdrop_stage
    global stage4_poinsettia_stage
    global stage4_cyclamen_stage
    global stage4_score
    global stage4_locked_to_pot
    global stage4_locked_pot_key
    global stage4_active_pot_key
    global stage4_message
    global stage4_completed
    global stage4_completion_time
    global active_character
    global last_sun_move_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    global cloud_activation_time
    stage4_sun_current_x = float(STAGE4_SUN_START_X)
    stage4_sun_current_y = float(STAGE4_SUN_START_Y)
    stage4_sun_target_x = float(STAGE4_SUN_START_X)
    stage4_sun_target_y = float(STAGE4_SUN_START_Y)
    stage4_winter_rose_stage = 0
    stage4_snowdrop_stage = 0
    stage4_poinsettia_stage = 0
    stage4_cyclamen_stage = 0
    stage4_score = 0
    stage4_locked_to_pot = False
    stage4_locked_pot_key = None
    stage4_active_pot_key = None
    stage4_message = "Move the sun from the lower gate to one of the winter pots."
    stage4_completed = False
    stage4_completion_time = None
    active_character = "sun"
    last_sun_move_time = 0
    rain_effect_start_time = 0
    rain_effect_x = float(STAGE4_SUN_START_X)
    rain_effect_y = float(STAGE4_SUN_START_Y)
    cloud_activation_time = 0.0
    clear_all_movement_holds()
    reset_stage4_chin_progress()
    reset_stage4_shoulder_progress()
    reset_stage4_retraction_progress()
    reset_stage4_rain_sequence()


def start_stage4_after_calibration():
    global game_state
    global mouse_left_clicked
    global current_stage_number
    global win_message
    reset_stage4_state_keep_calibration()
    current_stage_number = 4
    win_message = ""
    mouse_left_clicked = False
    start_new_session_metrics("stage_4_play", stage_number=4)
    game_state = "stage4"


def start_stage4_rain_sequence(pot_key):
    global stage4_rain_sequence_active
    global stage4_rain_pot_key
    global stage4_rain_start_time
    global stage4_stage3_pause_active
    global stage4_stage3_pause_pot_key
    global stage4_stage3_pause_start_time
    global rain_effect_start_time
    global rain_effect_x
    global rain_effect_y
    now = time.time()
    stage4_rain_sequence_active = True
    stage4_rain_pot_key = pot_key
    stage4_rain_start_time = now
    stage4_stage3_pause_active = False
    stage4_stage3_pause_pot_key = None
    stage4_stage3_pause_start_time = 0.0
    rain_effect_start_time = now
    rain_effect_x = float(stage4_sun_target_x)
    rain_effect_y = float(stage4_sun_target_y)


def update_stage4_rain_sequence():
    global stage4_rain_sequence_active
    global stage4_rain_pot_key
    global stage4_rain_start_time
    global stage4_stage3_pause_active
    global stage4_stage3_pause_pot_key
    global stage4_stage3_pause_start_time
    global stage4_score
    global stage4_completed
    global stage4_completion_time
    global game_state
    global win_message
    now = time.time()
    if stage4_rain_sequence_active:
        pot_key = stage4_rain_pot_key
        elapsed = now - stage4_rain_start_time
        if elapsed < STAGE4_RAIN_DURATION:
            return f"Rain is falling: {elapsed:.1f}s / {STAGE4_RAIN_DURATION:.1f}s"
        if get_stage4_pot_stage(pot_key) == 2:
            set_stage4_pot_stage(pot_key, 3)
            stage4_score = min(STAGE4_TOTAL_POTS, stage4_score + 1)
        stage4_rain_sequence_active = False
        stage4_rain_pot_key = None
        stage4_rain_start_time = 0.0
        stage4_stage3_pause_active = True
        stage4_stage3_pause_pot_key = pot_key
        stage4_stage3_pause_start_time = now
        return f"{get_stage4_pot_name(pot_key)} is fully grown. Wait a moment."
    if stage4_stage3_pause_active:
        elapsed = now - stage4_stage3_pause_start_time
        if elapsed < STAGE4_RETURN_DELAY_AFTER_POT:
            remaining = max(0.0, STAGE4_RETURN_DELAY_AFTER_POT - elapsed)
            return f"The winter flower is complete. Returning to the lower gate in {remaining:.1f}s."
        reset_stage4_rain_sequence()
        unlock_stage4_character_to_entrance()
        if all_stage4_pots_fully_grown():
            stage4_completed = True
            stage4_completion_time = now
            return f"Stage 4 complete! Win screen in {STAGE4_WIN_DELAY:.1f}s."
        return "Good job. Move to another winter pot."
    if stage4_completed and stage4_completion_time is not None:
        elapsed = now - stage4_completion_time
        if elapsed >= STAGE4_WIN_DELAY:
            mark_stage_completed(4)
            finalize_session_save("completed")
            win_message = "Stage 4 complete. All four winter flowers are fully grown."
            game_state = "win"
            return "Stage 4 complete. You won!"
        return f"Stage 4 complete! Win screen in {max(0.0, STAGE4_WIN_DELAY - elapsed):.1f}s."
    return None


def process_stage4_shoulder_to_cloud(current_pitch, current_yaw, current_roll, current_shoulder_features, current_shoulder_meta):
    global stage4_shoulder_total_time
    global stage4_shoulder_last_update_time
    global active_character
    global sun_shining_start_time
    global rain_effect_start_time
    global cloud_activation_time

    required_time = STAGE4_SHOULDER_REQUIRED_HOLD_TIME

    if active_character == "cloud":
        reset_stage4_shoulder_progress()
        return "Cloud is ready. Now move shoulders back for rain."

    if not (
        current_shoulder_features is not None and current_shoulder_meta is not None and
        shoulder_neutral_features is not None and shoulder_target_features is not None and
        shoulder_neutral_width is not None and shoulder_neutral_nose_y is not None and
        shoulder_neutral_angle is not None
    ):
        pause_stage4_shoulder_progress()
        return f"Lift shoulders. Progress saved: {stage4_shoulder_total_time:.1f}s / {required_time:.1f}s"

    head_ready = True
    if current_pitch is not None and neutral_pitch is not None and current_yaw is not None and neutral_yaw is not None and current_roll is not None and neutral_roll is not None:
        head_ready = (
            abs(angle_diff(current_pitch, neutral_pitch)) <= SHOULDER_HEAD_PITCH_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_yaw, neutral_yaw)) <= SHOULDER_HEAD_YAW_LIMIT_FOR_TOGGLE and
            abs(angle_diff(current_roll, neutral_roll)) <= SHOULDER_HEAD_ROLL_LIMIT_FOR_TOGGLE
        )
    if not head_ready:
        pause_stage4_shoulder_progress()
        return f"Keep head steady. Progress saved: {stage4_shoulder_total_time:.1f}s / {required_time:.1f}s"

    sh_progress, sh_side_error, sh_direct_error, sh_target_strength, sh_current_strength = shoulder_lift_metrics(
        current_shoulder_features,
        shoulder_neutral_features,
        shoulder_target_features
    )
    target_left = shoulder_target_features[0] - shoulder_neutral_features[0]
    target_right = shoulder_target_features[1] - shoulder_neutral_features[1]
    current_left = current_shoulder_features[0] - shoulder_neutral_features[0]
    current_right = current_shoulder_features[1] - shoulder_neutral_features[1]

    continuing = stage4_shoulder_total_time > 0.0 or stage4_shoulder_last_update_time is not None
    ratio = 0.35 if not continuing else 0.20
    weak_ratio = 0.18
    safe_left = max(abs(target_left), SHOULDER_MIN_SINGLE_LIFT)
    safe_right = max(abs(target_right), SHOULDER_MIN_SINGLE_LIFT)
    left_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_left * ratio)
    right_req = max(SHOULDER_MIN_SINGLE_LIFT * (1.0 if not continuing else 0.75), safe_right * ratio)
    left_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_left * weak_ratio)
    right_weak = max(SHOULDER_MIN_SINGLE_LIFT * 0.5, safe_right * weak_ratio)

    detected = sh_target_strength >= SHOULDER_MIN_TARGET_STRENGTH and (
        (current_left >= left_req and current_right >= right_req) or
        (current_left >= left_req and current_right >= right_weak) or
        (current_right >= right_req and current_left >= left_weak)
    )

    stage4_shoulder_total_time, stage4_shoulder_last_update_time = update_cumulative_hold_progress(
        detected,
        stage4_shoulder_total_time,
        stage4_shoulder_last_update_time,
        required_time
    )

    if stage4_shoulder_total_time >= required_time:
        active_character = "cloud"
        sun_shining_start_time = 0
        rain_effect_start_time = 0
        cloud_activation_time = time.time()
        clear_chin_histories()
        reset_stage4_shoulder_progress()
        return "Cloud activated! Now move shoulders back for 10 seconds."

    if detected:
        return f"Scapular Elevation total: {stage4_shoulder_total_time:.1f}s / {required_time:.1f}s"

    return (
        f"Scapular Elevation paused; progress saved {stage4_shoulder_total_time:.1f}s/{required_time:.1f}s | "
        f"L {current_left:.3f}/{left_req:.3f} | R {current_right:.3f}/{right_req:.3f}"
    )



def process_stage4_cloud_retraction_rain(current_features, face_detected, palms_detected, shoulders_detected, hands_outside_shoulders, shoulder_gate_info):
    global stage4_retraction_total_time
    global stage4_retraction_last_update_time

    required_time = STAGE4_RETRACTION_REQUIRED_HOLD_TIME

    if not (retraction_neutral_features is not None and retraction_target_features is not None and retraction_calibration_success):
        pause_stage4_retraction_progress()
        return f"Rain movement is not calibrated. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not face_detected:
        pause_stage4_retraction_progress()
        return f"Show your face. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not palms_detected:
        pause_stage4_retraction_progress()
        return f"Show both palms. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not shoulders_detected:
        pause_stage4_retraction_progress()
        return f"Shoulders not detected. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"
    if not hands_outside_shoulders:
        pause_stage4_retraction_progress()
        return f"Keep palms outside shoulder width. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"
    if current_features is None:
        pause_stage4_retraction_progress()
        return f"Keep face, shoulders, and palms visible. Progress saved: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"

    averaged = average_retraction_vectors(retraction_current_buffer)
    if averaged is None:
        averaged = current_features.copy()
    detected, score_value, target_strength, current_strength, progress, side_error = is_retraction(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )

    stage4_retraction_total_time, stage4_retraction_last_update_time = update_cumulative_hold_progress(
        detected,
        stage4_retraction_total_time,
        stage4_retraction_last_update_time,
        required_time
    )

    if stage4_retraction_total_time >= required_time:
        start_stage4_rain_sequence(stage4_locked_pot_key)
        reset_stage4_retraction_progress()
        return f"Rain started above {get_stage4_pot_name(stage4_locked_pot_key)}!"

    if detected:
        return f"Scapular Retraction total: {stage4_retraction_total_time:.1f}s / {required_time:.1f}s"

    strict_ok, gate_info, req_gap, req_left, req_right = strict_retraction_gate(
        averaged,
        retraction_neutral_features,
        retraction_target_features
    )
    return (
        f"Retraction paused; progress saved {stage4_retraction_total_time:.1f}s/{required_time:.1f}s | "
        f"gap {gate_info['gap_delta']:.2f}/{req_gap:.2f} | "
        f"L {gate_info['left_delta']:.2f}/{req_left:.2f} | R {gate_info['right_delta']:.2f}/{req_right:.2f}"
    )



def _apply_stage4_move(candidate_x, candidate_y):
    global stage4_sun_target_x
    global stage4_sun_target_y
    global last_sun_move_time
    if not can_stage4_move_to(candidate_x, candidate_y):
        return False, "Cannot move outside the game area."
    stage4_sun_target_x = float(candidate_x)
    stage4_sun_target_y = float(candidate_y)
    last_sun_move_time = time.time()
    pot_message = check_stage4_pot_reached()
    return True, pot_message if pot_message else "Good movement. Move freely toward a winter pot."


def process_stage4_free_movement(current_pitch, current_yaw, current_side_bend_angle):
    global flexion_hold_start
    global extension_hold_start
    global left_side_bend_hold_start
    global right_side_bend_hold_start
    now = time.time()
    if now - last_sun_move_time < STAGE4_MOVE_COOLDOWN:
        return "Move the sun from the lower gate to an unfinished winter pot."

    # Horizontal movement: left/right side bend.
    if current_side_bend_angle is not None and neutral_side_bend_angle is not None and left_side_bend_direction is not None and right_side_bend_direction is not None:
        delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
        left_amount = left_side_bend_direction * delta
        right_amount = right_side_bend_direction * delta
        left_detected = False
        right_detected = False
        if left_side_bend_threshold is not None and left_amount >= left_side_bend_threshold:
            if left_side_bend_hold_start is None:
                left_side_bend_hold_start = now
            left_detected = now - left_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME
        else:
            left_side_bend_hold_start = None
        if right_side_bend_threshold is not None and right_amount >= right_side_bend_threshold:
            if right_side_bend_hold_start is None:
                right_side_bend_hold_start = now
            right_detected = now - right_side_bend_hold_start >= SIDE_BEND_REQUIRED_HOLD_TIME
        else:
            right_side_bend_hold_start = None
        if left_detected or right_detected:
            direction = 1 if right_detected else -1
            candidate_x = stage4_sun_target_x + direction * STAGE4_MOVE_DISTANCE
            candidate_x = max(STAGE4_SUN_MIN_X, min(STAGE4_SUN_MAX_X, candidate_x))
            moved, message = _apply_stage4_move(candidate_x, stage4_sun_target_y)
            left_side_bend_hold_start = None
            right_side_bend_hold_start = None
            return message

    # Vertical movement: Flexion moves forward/down; Extension moves backward/up.
    if current_pitch is not None and neutral_pitch is not None and flexion_direction is not None and extension_direction is not None:
        pitch_delta = angle_diff(current_pitch, neutral_pitch)
        yaw_delta = abs(angle_diff(current_yaw, neutral_yaw)) if current_yaw is not None and neutral_yaw is not None else 0.0
        if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
            flexion_hold_start = None
            extension_hold_start = None
            return "Keep your face forward while moving."
        flexion_amount = flexion_direction * pitch_delta
        extension_amount = extension_direction * pitch_delta
        flexion_detected = False
        extension_detected = False
        if flexion_threshold is not None and flexion_amount >= flexion_threshold:
            if flexion_hold_start is None:
                flexion_hold_start = now
            flexion_detected = now - flexion_hold_start >= FLEXION_REQUIRED_HOLD_TIME
        else:
            flexion_hold_start = None
        if extension_threshold is not None and extension_amount >= extension_threshold:
            if extension_hold_start is None:
                extension_hold_start = now
            extension_detected = now - extension_hold_start >= EXTENSION_REQUIRED_HOLD_TIME
        else:
            extension_hold_start = None
        if flexion_detected or extension_detected:
            direction = 1 if flexion_detected else -1
            candidate_y = stage4_sun_target_y + direction * STAGE4_MOVE_DISTANCE
            candidate_y = max(STAGE4_SUN_MIN_Y, min(STAGE4_SUN_MAX_Y, candidate_y))
            moved, message = _apply_stage4_move(stage4_sun_target_x, candidate_y)
            flexion_hold_start = None
            extension_hold_start = None
            return message

    return "Move the sun from the lower gate to an unfinished winter pot."


def process_stage4_stage(current_pitch, current_yaw, current_roll, current_side_bend_angle, current_chin_features, current_shoulder_features, current_shoulder_meta, current_retraction_features, retraction_face_detected, retraction_palms_detected, retraction_shoulders_detected, retraction_hands_outside_shoulders, retraction_shoulder_gate_info):
    global stage4_message
    global sun_shining_start_time
    rain_status = update_stage4_rain_sequence()
    if rain_status is not None:
        stage4_message = rain_status
    elif stage4_completed:
        stage4_message = "Stage 4 complete. Great winter work!"
    elif stage4_locked_to_pot:
        pot_stage = get_stage4_pot_stage(stage4_locked_pot_key)
        pot_name = get_stage4_pot_name(stage4_locked_pot_key)
        if pot_stage == 1:
            reset_stage4_shoulder_progress()
            reset_stage4_retraction_progress()
            if current_chin_features is not None and chin_neutral_features is not None and chin_target_features is not None:
                is_chin, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                    current_chin_features,
                    chin_neutral_features,
                    chin_target_features,
                    current_pitch,
                    neutral_pitch,
                    current_yaw,
                    neutral_yaw,
                    current_roll,
                    neutral_roll
                )
                total = update_stage4_chin_tuck_progress(is_chin)
                if total >= STAGE4_CHIN_REQUIRED_TOTAL_TIME:
                    set_stage4_pot_stage(stage4_locked_pot_key, 2)
                    reset_stage4_chin_progress()
                    sun_shining_start_time = time.time()
                    stage4_message = (
                        f"Chin Tuck complete! The ice on {pot_name} melted and it is Stage 2. "
                        "Lift shoulders for 5 seconds."
                    )
                elif is_chin:
                    stage4_message = f"Hold Chin Tuck: {total:.1f}s / {STAGE4_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
                else:
                    stage4_message = f"Hold Chin Tuck for 10 seconds. Progress: {total:.1f}s / {STAGE4_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
            else:
                update_stage4_chin_tuck_progress(False)
                stage4_message = "Keep your face visible, then hold Chin Tuck."
        elif pot_stage == 2:
            reset_stage4_chin_progress()
            if active_character == "sun":
                stage4_message = process_stage4_shoulder_to_cloud(
                    current_pitch,
                    current_yaw,
                    current_roll,
                    current_shoulder_features,
                    current_shoulder_meta
                )
            else:
                reset_stage4_shoulder_progress()
                stage4_message = process_stage4_cloud_retraction_rain(
                    current_retraction_features,
                    retraction_face_detected,
                    retraction_palms_detected,
                    retraction_shoulders_detected,
                    retraction_hands_outside_shoulders,
                    retraction_shoulder_gate_info
                )
        else:
            stage4_message = "This flower is complete. Returning to the lower gate."
    else:
        reset_stage4_chin_progress()
        reset_stage4_shoulder_progress()
        reset_stage4_retraction_progress()
        stage4_message = process_stage4_free_movement(current_pitch, current_yaw, current_side_bend_angle)
    return stage4_message


def get_stage4_plant_asset(pot_key, stage):
    groups = {
        "winter_rose": (
            stage4_winter_rose_stage1_img,
            stage4_winter_rose_stage2_img,
            stage4_winter_rose_stage3_img,
        ),
        "snowdrop": (
            stage4_snowdrop_stage1_img,
            stage4_snowdrop_stage2_img,
            stage4_snowdrop_stage3_img,
        ),
        "poinsettia": (
            stage4_poinsettia_stage1_img,
            stage4_poinsettia_stage2_img,
            stage4_poinsettia_stage3_img,
        ),
        "cyclamen": (
            stage4_cyclamen_stage1_img,
            stage4_cyclamen_stage2_img,
            stage4_cyclamen_stage3_img,
        ),
    }
    assets = groups.get(pot_key)
    if assets is None or stage <= 0:
        return None
    return assets[min(2, stage - 1)]


def draw_stage4_plant_fallback(frame, pot_key, stage):
    if stage <= 0:
        return frame
    pot_x, pot_y = get_stage4_pot_position(pot_key)
    colors = {
        "winter_rose": ((80, 130, 92), (65, 70, 220)),
        "snowdrop": ((90, 145, 95), (245, 245, 245)),
        "poinsettia": ((70, 125, 80), (45, 55, 230)),
        "cyclamen": ((80, 140, 90), (205, 90, 230)),
    }
    leaf_color, flower_color = colors.get(pot_key, ((80, 140, 90), (220, 220, 245)))
    radius = 22 + stage * 13
    center_y = int(pot_y - 48)
    cv2.circle(frame, (int(pot_x), center_y), radius, leaf_color, -1, cv2.LINE_AA)
    for angle in range(0, 360, max(38, 85 - stage * 15)):
        rad = math.radians(angle)
        fx = int(pot_x + math.cos(rad) * radius * 0.75)
        fy = int(center_y + math.sin(rad) * radius * 0.55)
        cv2.circle(frame, (fx, fy), 5 + stage, flower_color, -1, cv2.LINE_AA)
    if stage == 1:
        cv2.circle(frame, (int(pot_x), center_y), radius + 5, (245, 235, 210), 3, cv2.LINE_AA)
    return frame


def draw_stage4_plant_on_pot(frame, pot_key, stage):
    if stage <= 0:
        return frame
    asset = get_stage4_plant_asset(pot_key, stage)
    if asset is None:
        return draw_stage4_plant_fallback(frame, pot_key, stage)
    pot_x, pot_y = get_stage4_pot_position(pot_key)
    factor = 0.68 if stage == 1 else (0.86 if stage == 2 else 1.0)
    desired_size = max(20, int(STAGE4_PLANT_SIZE * factor))
    resized = cv2.resize(asset, (desired_size, desired_size), interpolation=cv2.INTER_AREA)
    x = int(pot_x - desired_size / 2)
    y = int(pot_y - desired_size + 18)
    return overlay_transparent(frame, resized, x, y)


def draw_stage4_scene_base():
    if stage4_background_img is not None:
        frame = stage4_background_img.copy()
    else:
        frame = background.copy()
    frame = draw_stage4_plant_on_pot(frame, "winter_rose", stage4_winter_rose_stage)
    frame = draw_stage4_plant_on_pot(frame, "snowdrop", stage4_snowdrop_stage)
    frame = draw_stage4_plant_on_pot(frame, "poinsettia", stage4_poinsettia_stage)
    frame = draw_stage4_plant_on_pot(frame, "cyclamen", stage4_cyclamen_stage)
    return frame


def draw_stage4_instruction_card(frame):
    if stage4_completed:
        title = "Stage 4 Complete"
    elif not stage4_locked_to_pot:
        title = "Stage 4 - Winter Garden"
    elif get_stage4_pot_stage(stage4_locked_pot_key) == 1:
        title = "Step 1 - Thaw with Chin Tuck"
    elif get_stage4_pot_stage(stage4_locked_pot_key) == 2 and active_character == "sun":
        title = "Step 2 - Lift Shoulders"
    else:
        title = "Step 3 - Make Rain"

    # Move the message card to the bottom when the top flowers are active.
    if stage4_locked_to_pot and stage4_locked_pot_key in {"winter_rose", "snowdrop"}:
        x1, y1, x2, y2 = 255, 612, 1025, 698
    else:
        x1, y1, x2, y2 = 255, 20, 1025, 106
    draw_transparent_rounded_rect(frame, x1 + 5, y1 + 6, x2 + 5, y2 + 6, (45, 60, 80), alpha=0.20, radius=28)
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (245, 252, 255), alpha=0.92, radius=28)
    draw_rounded_rect(frame, x1, y1, x2, y2, (170, 120, 80), radius=28, thickness=3)
    draw_centered_text(frame, title, WIDTH // 2, y1 + 35, scale=0.70, color=(145, 85, 45), thickness=2)
    lines = wrap_text_lines(stage4_message, x2 - x1 - 80, scale=0.40, thickness=1)
    msg_y = y1 + 64
    for line in lines[:2]:
        draw_centered_text(frame, line, WIDTH // 2, msg_y, scale=0.40, color=(55, 65, 75), thickness=1)
        msg_y += 19


def draw_stage4_progress_card(frame):
    x1, y1, x2, y2 = 24, 22, 236, 108
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (242, 250, 255), alpha=0.92, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (170, 120, 80), radius=24, thickness=3)
    draw_ui_text(frame, "Winter Garden", x1 + 20, y1 + 33, scale=0.49, color=(135, 80, 45), thickness=2)
    draw_ui_text(frame, f"Score: {stage4_score} / {STAGE4_TOTAL_POTS}", x1 + 20, y1 + 65, scale=0.47, color=(55, 65, 75), thickness=2)
    completed = [
        stage4_winter_rose_stage,
        stage4_snowdrop_stage,
        stage4_poinsettia_stage,
        stage4_cyclamen_stage,
    ]
    for i, plant_stage in enumerate(completed):
        if plant_stage == 3:
            draw_check_icon(frame, x2 - 92 + i * 23, y1 + 62, radius=9)


def draw_stage4_screen(current_sun_frame, current_cloud_frame):
    global stage4_sun_current_x
    global stage4_sun_current_y
    frame = draw_stage4_scene_base()
    if (time.time() - rain_effect_start_time) <= STAGE4_RAIN_DURATION:
        frame = draw_rain(frame, int(rain_effect_x), int(rain_effect_y), SUN_SIZE)
    stage4_sun_current_x += (stage4_sun_target_x - stage4_sun_current_x) * STAGE4_VISUAL_SMOOTHING_FACTOR
    stage4_sun_current_y += (stage4_sun_target_y - stage4_sun_current_y) * STAGE4_VISUAL_SMOOTHING_FACTOR
    sun_is_shining = (
        active_character == "sun" and
        (stage4_chin_tuck_last_update_time is not None or (time.time() - sun_shining_start_time) <= SUN_SHINING_DURATION)
    )
    if active_character == "sun":
        if sun_is_shining:
            frame = draw_sun_glow(frame, int(stage4_sun_current_x), int(stage4_sun_current_y), SUN_SIZE)
        character_frame = current_sun_frame
    else:
        character_frame = current_cloud_frame
    frame = overlay_transparent(frame, character_frame, int(stage4_sun_current_x), int(stage4_sun_current_y))
    draw_stage4_instruction_card(frame)
    draw_stage4_progress_card(frame)
    home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
    return draw_home_icon_button(frame, hovered=home_hovered)


def reset_full_game_to_initial_state():
    global neutral_pitch, neutral_yaw, neutral_roll, neutral_side_bend_angle
    global flexion_direction, flexion_threshold, extension_direction, extension_threshold
    global left_side_bend_direction, left_side_bend_threshold, right_side_bend_direction, right_side_bend_threshold
    global smoothed_pitch, smoothed_yaw, smoothed_roll
    global flexion_hold_start, extension_hold_start, left_side_bend_hold_start, right_side_bend_hold_start
    global sun_current_x, sun_current_y, sun_target_x, sun_target_y
    global last_sun_move_time, sun_shining_start_time, rain_effect_start_time, rain_effect_x, rain_effect_y, cloud_activation_time
    global active_character, score
    global top_flower_stage, bottom_flower_stage, right_orchid_stage, south_east_bluebloom_stage, left_tulip_stage, south_west_peony_stage
    global top_flower_animating, bottom_flower_animating, right_orchid_animating, south_east_bluebloom_animating, left_tulip_animating, south_west_peony_animating
    global top_flower_start_time, bottom_flower_start_time, right_orchid_start_time, south_east_bluebloom_start_time, left_tulip_start_time, south_west_peony_start_time
    global active_flower, character_locked_to_flower, locked_flower_key, reached_side_pots
    global chin_neutral_features, chin_target_features, chin_neutral_pitch, chin_neutral_yaw, chin_neutral_eye_roll, chin_neutral_face_width
    global smoothed_chin_eye_roll, chin_tuck_hold_start, stage_chin_hold_start, rain_chin_hold_start, stage_chin_last_seen_time, rain_chin_last_seen_time, rain_waiting_for_chin_release
    global shoulder_neutral_features, shoulder_target_features, shoulder_neutral_nose_y, shoulder_neutral_width, shoulder_neutral_angle
    global smoothed_shoulder_features, smoothed_shoulder_nose_y, smoothed_shoulder_width, smoothed_shoulder_angle
    global shoulder_hold_start, shoulder_release_start_time, shoulder_toggle_waiting_release
    global game_state, game_finished, win_message, mouse_left_clicked, calibration_return_mode, pause_menu_enter_time, pause_return_state

    neutral_pitch = None
    neutral_yaw = None
    neutral_roll = None
    neutral_side_bend_angle = None
    flexion_direction = None
    flexion_threshold = None
    extension_direction = None
    extension_threshold = None
    left_side_bend_direction = None
    left_side_bend_threshold = None
    right_side_bend_direction = None
    right_side_bend_threshold = None
    smoothed_pitch = None
    smoothed_yaw = None
    smoothed_roll = None
    flexion_hold_start = None
    extension_hold_start = None
    left_side_bend_hold_start = None
    right_side_bend_hold_start = None

    sun_current_x = float(sun_x)
    sun_current_y = float(sun_y)
    sun_target_x = float(sun_x)
    sun_target_y = float(sun_y)

    last_sun_move_time = 0
    sun_shining_start_time = 0
    rain_effect_start_time = 0
    rain_effect_x = float(sun_x)
    rain_effect_y = float(sun_y)
    cloud_activation_time = 0.0
    active_character = "sun"
    score = 0

    top_flower_stage = 0
    bottom_flower_stage = 0
    right_orchid_stage = 0
    south_east_bluebloom_stage = 0
    left_tulip_stage = 0
    south_west_peony_stage = 0
    top_flower_animating = False
    bottom_flower_animating = False
    right_orchid_animating = False
    south_east_bluebloom_animating = False
    left_tulip_animating = False
    south_west_peony_animating = False
    top_flower_start_time = 0
    bottom_flower_start_time = 0
    right_orchid_start_time = 0
    south_east_bluebloom_start_time = 0
    left_tulip_start_time = 0
    south_west_peony_start_time = 0

    active_flower = None
    character_locked_to_flower = False
    locked_flower_key = None
    reached_side_pots = set()

    chin_neutral_features = None
    chin_target_features = None
    chin_neutral_pitch = None
    chin_neutral_yaw = None
    chin_neutral_eye_roll = None
    chin_neutral_face_width = None
    smoothed_chin_eye_roll = None
    clear_chin_histories()
    chin_tuck_hold_start = None
    stage_chin_hold_start = None
    rain_chin_hold_start = None
    stage_chin_last_seen_time = None
    rain_chin_last_seen_time = None
    rain_waiting_for_chin_release = False

    shoulder_neutral_features = None
    shoulder_target_features = None
    shoulder_neutral_nose_y = None
    shoulder_neutral_width = None
    shoulder_neutral_angle = None
    smoothed_shoulder_features = None
    smoothed_shoulder_nose_y = None
    smoothed_shoulder_width = None
    smoothed_shoulder_angle = None
    clear_shoulder_histories()
    shoulder_hold_start = None
    shoulder_release_start_time = None
    shoulder_toggle_waiting_release = False

    reset_locked_chin_tuck_progress()
    reset_locked_shoulder_lift_progress()
    reset_retraction_calibration_state(clear_saved=True)
    reset_locked_retraction_progress()
    reset_locked_rain_sequence()
    reset_stage3_state_keep_calibration()
    reset_stage4_state_keep_calibration()

    game_state = "calibration"
    game_finished = False
    win_message = ""
    mouse_left_clicked = False
    calibration_return_mode = "new_game"
    pause_menu_enter_time = None
    pause_return_state = "game"


# start_new_game_from_main_menu is defined above with the stage-select flow.


def session_empty_pose():
    return {
        "pitch_min": None, "pitch_max": None,
        "yaw_min": None, "yaw_max": None,
        "roll_min": None, "roll_max": None,
    }


def create_empty_movement_metrics():
    return {
        "flexion": {"label": "Flexion", "max_value": 0.0, "samples": 0},
        "extension": {"label": "Extension", "max_value": 0.0, "samples": 0},
        "left_bend": {"label": "Left Side Bend", "max_value": 0.0, "samples": 0},
        "right_bend": {"label": "Right Side Bend", "max_value": 0.0, "samples": 0},
        "chin_tuck": {"label": "Chin Tuck", "max_value": 0.0, "samples": 0},
        "shoulder_lift": {"label": "Scapular Elevation", "max_value": 0.0, "samples": 0},
        "palm_retraction": {"label": "Scapular Retraction", "max_value": 0.0, "samples": 0},
    }



def get_stage_runtime_snapshot(stage_number=None):
    if stage_number is None:
        stage_number = current_stage_number
    try:
        stage_number = int(stage_number)
    except Exception:
        stage_number = None
    stage_def = get_stage_definition(stage_number) if stage_number is not None else None
    stage_name = stage_def.get("title", f"Stage {stage_number}") if stage_def else "Unknown Stage"
    score_value = 0
    max_score = 0
    completed = False
    if stage_number == 1:
        max_score = 1
        completed = bool(tutorial_completed)
        score_value = 1 if completed else 0
    elif stage_number == 2:
        max_score = STAGE2_TOTAL_POTS
        score_value = int(stage2_score)
        completed = bool(stage2_completed or all_stage2_pots_fully_grown())
    elif stage_number == 3:
        max_score = STAGE3_TOTAL_POTS
        score_value = int(stage3_score)
        completed = bool(stage3_completed or all_stage3_pots_fully_grown())
    elif stage_number == 4:
        max_score = STAGE4_TOTAL_POTS
        score_value = int(stage4_score)
        completed = bool(stage4_completed or all_stage4_pots_fully_grown())
    elif stage_number == 5:
        max_score = TOTAL_FLOWERS
        score_value = int(score)
        completed = bool(game_finished or all_flowers_fully_grown())
    return {
        "stage_number": stage_number,
        "stage_name": stage_name,
        "score": score_value,
        "max_score": max_score,
        "completed": completed,
    }


def start_new_session_metrics(reason="new_session", stage_number=None):
    global current_session
    global session_saved
    snapshot = get_stage_runtime_snapshot(stage_number) if stage_number is not None else {
        "stage_number": None,
        "stage_name": "Calibration",
    }
    now = time.time()
    current_session = {
        "active": True,
        "saved": False,
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "start_datetime": datetime.now().isoformat(timespec="seconds"),
        "start_time": now,
        "reason": reason,
        "stage_number": snapshot.get("stage_number"),
        "stage_name": snapshot.get("stage_name", "Unknown Stage"),
        "paused_total": 0.0,
        "pause_started_at": None,
        "pose": session_empty_pose(),
        "movements": create_empty_movement_metrics(),
        "angle_samples": [],
        "last_sample_time": 0.0,
    }
    session_saved = False


def get_current_session_elapsed(now=None):
    if current_session is None:
        return 0.0
    if now is None:
        now = time.time()
    start_time = float(current_session.get("start_time", now))
    paused_total = float(current_session.get("paused_total", 0.0))
    pause_started = current_session.get("pause_started_at")
    active_pause = max(0.0, now - pause_started) if pause_started is not None else 0.0
    return max(0.0, now - start_time - paused_total - active_pause)


def pause_current_session_clock():
    if current_session is not None and current_session.get("active", False):
        if current_session.get("pause_started_at") is None:
            current_session["pause_started_at"] = time.time()


def resume_current_session_clock():
    if current_session is not None and current_session.get("active", False):
        pause_started = current_session.get("pause_started_at")
        if pause_started is not None:
            current_session["paused_total"] = float(current_session.get("paused_total", 0.0)) + max(0.0, time.time() - pause_started)
            current_session["pause_started_at"] = None


def update_pose_range(pose_dict, pitch, yaw, roll):
    if pitch is not None:
        pose_dict["pitch_min"] = pitch if pose_dict["pitch_min"] is None else min(pose_dict["pitch_min"], pitch)
        pose_dict["pitch_max"] = pitch if pose_dict["pitch_max"] is None else max(pose_dict["pitch_max"], pitch)
    if yaw is not None:
        pose_dict["yaw_min"] = yaw if pose_dict["yaw_min"] is None else min(pose_dict["yaw_min"], yaw)
        pose_dict["yaw_max"] = yaw if pose_dict["yaw_max"] is None else max(pose_dict["yaw_max"], yaw)
    if roll is not None:
        pose_dict["roll_min"] = roll if pose_dict["roll_min"] is None else min(pose_dict["roll_min"], roll)
        pose_dict["roll_max"] = roll if pose_dict["roll_max"] is None else max(pose_dict["roll_max"], roll)


def update_movement_metric(name, value):
    if current_session is None or not current_session.get("active", False):
        return
    if name not in current_session["movements"]:
        return
    try:
        value = float(value)
    except Exception:
        return
    if value < 0:
        return
    current_session["movements"][name]["samples"] += 1
    if value > current_session["movements"][name]["max_value"]:
        current_session["movements"][name]["max_value"] = value


def get_analysis_active_angle_threshold(movement_key):
    """
    Minimum angle that counts as an active movement sample for the Analysis screen.
    It uses the user's own calibration threshold when available, with a safe minimum.
    """
    try:
        if movement_key == "flexion" and flexion_threshold is not None:
            return max(ANALYSIS_MIN_ACTIVE_ANGLE_DEG, float(flexion_threshold) * 0.35)
        if movement_key == "extension" and extension_threshold is not None:
            return max(ANALYSIS_MIN_ACTIVE_ANGLE_DEG, float(extension_threshold) * 0.35)
        if movement_key == "left_bend" and left_side_bend_threshold is not None:
            return max(ANALYSIS_MIN_ACTIVE_ANGLE_DEG, float(left_side_bend_threshold) * 0.35)
        if movement_key == "right_bend" and right_side_bend_threshold is not None:
            return max(ANALYSIS_MIN_ACTIVE_ANGLE_DEG, float(right_side_bend_threshold) * 0.35)
    except Exception:
        pass
    return ANALYSIS_MIN_ACTIVE_ANGLE_DEG


def add_calibration_analysis_sample(movement_key, metric_value, calibration_error, is_active, is_correct):
    """
    Stores one quality sample for the Analysis page.
    For ROM movements, metric_value is angle in degrees and calibration_error is
    the unwanted posture-plane deviation in degrees.
    For non-ROM movements, metric_value is calibration progress percent and
    calibration_error is a normalized match-error percentage.
    """
    if current_session is None or not current_session.get("active", False):
        return
    if movement_key not in ANALYSIS_ALL_MOVEMENT_KEYS:
        return
    if game_state not in {"tutorial", "stage2", "stage3", "stage4", "game"}:
        return

    try:
        metric_value = max(0.0, float(metric_value))
        calibration_error = max(0.0, float(calibration_error))
    except Exception:
        return

    current_session.setdefault("angle_samples", []).append({
        "movement": movement_key,
        "time_seconds": get_current_session_elapsed(),
        "angle": metric_value,
        "error_degree": calibration_error,
        "is_active": 1 if is_active else 0,
        "is_correct": 1 if is_correct else 0,
    })


def add_angle_analysis_sample(movement_key, angle_value, error_degree):
    """
    ROM helper kept for the angle-time chart.
    Correct/Error is calculated only when the movement is active enough.
    """
    if movement_key not in ANALYSIS_ROM_MOVEMENT_KEYS:
        return

    try:
        angle_value = max(0.0, float(angle_value))
        error_degree = max(0.0, float(error_degree))
    except Exception:
        return

    active_threshold = get_analysis_active_angle_threshold(movement_key)
    is_active = angle_value >= active_threshold
    is_correct = bool(is_active and error_degree <= ANALYSIS_ERROR_TOLERANCE_DEG)

    add_calibration_analysis_sample(
        movement_key,
        angle_value,
        error_degree,
        is_active,
        is_correct
    )


def update_session_frame_metrics(
    current_pitch,
    current_yaw,
    current_roll,
    current_side_bend_angle,
    current_chin_features,
    current_shoulder_features,
    current_retraction_features
):
    if current_session is None or not current_session.get("active", False):
        return

    now = time.time()
    if now - current_session.get("last_sample_time", 0.0) < ANALYSIS_SAMPLE_INTERVAL_SECONDS:
        return
    current_session["last_sample_time"] = now

    update_pose_range(current_session["pose"], current_pitch, current_yaw, current_roll)

    try:
        pitch_delta = None
        yaw_delta_abs = 0.0
        roll_delta_abs = 0.0

        if current_pitch is not None and neutral_pitch is not None:
            pitch_delta = angle_diff(current_pitch, neutral_pitch)
            if current_yaw is not None and neutral_yaw is not None:
                yaw_delta_abs = abs(angle_diff(current_yaw, neutral_yaw))
            if current_roll is not None and neutral_roll is not None:
                roll_delta_abs = abs(angle_diff(current_roll, neutral_roll))

            # For flexion/extension, the main angle is pitch; error is unwanted yaw/roll drift.
            flexion_angle = None
            extension_angle = None

            if flexion_direction is not None:
                flexion_angle = max(0.0, flexion_direction * pitch_delta)
                update_movement_metric("flexion", flexion_angle)

            if extension_direction is not None:
                extension_angle = max(0.0, extension_direction * pitch_delta)
                update_movement_metric("extension", extension_angle)

            pitch_plane_error = math.sqrt(yaw_delta_abs * yaw_delta_abs + roll_delta_abs * roll_delta_abs)

            if flexion_angle is not None:
                add_angle_analysis_sample("flexion", flexion_angle, pitch_plane_error)
            if extension_angle is not None:
                add_angle_analysis_sample("extension", extension_angle, pitch_plane_error)

        if current_side_bend_angle is not None and neutral_side_bend_angle is not None:
            side_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)

            # For left/right side bend, the main angle is eye-line/roll angle;
            # error is unwanted flexion/extension and face rotation drift.
            side_pitch_error = 0.0
            side_yaw_error = 0.0
            if current_pitch is not None and neutral_pitch is not None:
                side_pitch_error = abs(angle_diff(current_pitch, neutral_pitch))
            if current_yaw is not None and neutral_yaw is not None:
                side_yaw_error = abs(angle_diff(current_yaw, neutral_yaw))
            side_plane_error = math.sqrt(side_pitch_error * side_pitch_error + side_yaw_error * side_yaw_error)

            left_angle = None
            right_angle = None

            if left_side_bend_direction is not None:
                left_angle = max(0.0, left_side_bend_direction * side_delta)
                update_movement_metric("left_bend", left_angle)

            if right_side_bend_direction is not None:
                right_angle = max(0.0, right_side_bend_direction * side_delta)
                update_movement_metric("right_bend", right_angle)

            if left_angle is not None:
                add_angle_analysis_sample("left_bend", left_angle, side_plane_error)
            if right_angle is not None:
                add_angle_analysis_sample("right_bend", right_angle, side_plane_error)

        if current_chin_features is not None and chin_neutral_features is not None and chin_target_features is not None:
            is_chin, chin_score, target_strength, current_strength, progress, side_error = is_simple_chin_tuck(
                current_chin_features,
                chin_neutral_features,
                chin_target_features,
                current_pitch,
                neutral_pitch,
                current_yaw,
                neutral_yaw,
                current_roll,
                neutral_roll
            )
            if target_strength >= CHIN_MIN_TARGET_STRENGTH:
                metric_value = max(0.0, progress * 100.0)
                update_movement_metric("chin_tuck", metric_value)
                is_active = bool(
                    current_strength >= (target_strength * CHIN_ENOUGH_MOVEMENT_RATIO) or
                    progress >= (CHIN_PROGRESS_MIN * 0.60)
                )
                add_calibration_analysis_sample(
                    "chin_tuck",
                    metric_value,
                    min(999.0, chin_score * 100.0),
                    is_active,
                    bool(is_active and is_chin)
                )

        if current_shoulder_features is not None and shoulder_neutral_features is not None and shoulder_target_features is not None:
            progress, side_error, direct_error, target_strength, current_strength = shoulder_lift_metrics(current_shoulder_features, shoulder_neutral_features, shoulder_target_features)
            if target_strength >= SHOULDER_MIN_TARGET_STRENGTH:
                metric_value = max(0.0, progress * 100.0)
                update_movement_metric("shoulder_lift", metric_value)
                is_active = bool(
                    current_strength >= (target_strength * 0.20) or
                    progress >= SHOULDER_MIN_PROGRESS
                )
                is_correct = bool(
                    is_active and
                    (
                        direct_error <= SHOULDER_MATCH_THRESHOLD or
                        (progress >= SHOULDER_MIN_PROGRESS and side_error <= 1.25)
                    )
                )
                add_calibration_analysis_sample(
                    "shoulder_lift",
                    metric_value,
                    min(999.0, direct_error * 100.0),
                    is_active,
                    is_correct
                )

        if current_retraction_features is not None and retraction_neutral_features is not None and retraction_target_features is not None:
            detected, score_value, target_strength, current_strength, progress, side_error = is_retraction(
                current_retraction_features,
                retraction_neutral_features,
                retraction_target_features
            )
            if target_strength >= RETRACTION_MIN_TARGET_STRENGTH:
                metric_value = max(0.0, progress * 100.0)
                update_movement_metric("palm_retraction", metric_value)
                is_active = bool(
                    current_strength >= (target_strength * RETRACTION_ENOUGH_MOVEMENT_RATIO * 0.75) or
                    progress >= 0.25
                )
                is_correct = bool(
                    is_active and
                    (
                        detected or
                        (
                            score_value <= RETRACTION_MATCH_THRESHOLD and
                            RETRACTION_PROGRESS_MIN <= progress <= RETRACTION_PROGRESS_MAX and
                            side_error <= RETRACTION_SIDE_ERROR_MAX
                        )
                    )
                )
                add_calibration_analysis_sample(
                    "palm_retraction",
                    metric_value,
                    min(999.0, score_value * 100.0),
                    is_active,
                    is_correct
                )
    except Exception:
        pass



def ensure_csv_schema(csv_path, fieldnames):
    """Safely adds new columns while preserving every old CSV row."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        return
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            rows = list(reader)
        if old_fields == fieldnames:
            return
        temp_path = csv_path + ".schema_tmp"
        with open(temp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(temp_path, csv_path)
    except Exception as exc:
        print(f"Could not update CSV schema for {csv_path}: {exc}")


def save_session_angle_samples(profile_id, profile_name, session_id, session_datetime):
    if current_session is None:
        return
    samples = current_session.get("angle_samples", [])
    if len(samples) == 0:
        return
    ensure_data_dir()
    fieldnames = [
        "profile_id", "profile_name", "session_id", "datetime",
        "stage_number", "stage_name",
        "movement", "movement_label", "time_seconds", "angle",
        "error_degree", "is_active", "is_correct"
    ]
    ensure_csv_schema(SESSION_ANALYSIS_CSV_PATH, fieldnames)
    file_exists = os.path.exists(SESSION_ANALYSIS_CSV_PATH) and os.path.getsize(SESSION_ANALYSIS_CSV_PATH) > 0
    with open(SESSION_ANALYSIS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for sample in samples:
            movement_key = sample.get("movement", "")
            if movement_key not in ANALYSIS_ALL_MOVEMENT_KEYS:
                continue
            writer.writerow({
                "profile_id": profile_id,
                "profile_name": profile_name,
                "session_id": session_id,
                "datetime": session_datetime,
                "stage_number": current_session.get("stage_number", ""),
                "stage_name": current_session.get("stage_name", ""),
                "movement": movement_key,
                "movement_label": MOVEMENT_DISPLAY_NAMES.get(movement_key, movement_key),
                "time_seconds": f"{float(sample.get('time_seconds', 0.0)):.3f}",
                "angle": f"{float(sample.get('angle', 0.0)):.3f}",
                "error_degree": f"{float(sample.get('error_degree', 0.0)):.3f}",
                "is_active": int(sample.get("is_active", 0)),
                "is_correct": int(sample.get("is_correct", 0)),
            })



def finalize_session_save(result="exit"):
    global current_session
    global session_saved
    if current_session is None or not current_session.get("active", False):
        return
    if current_session.get("saved", False) or session_saved:
        return
    # Calibration/menu placeholders are intentionally not shown as played stages.
    stage_number = current_session.get("stage_number")
    if stage_number is None:
        current_session["saved"] = True
        current_session["active"] = False
        session_saved = True
        return
    ensure_data_dir()
    fieldnames = [
        "profile_id", "profile_name", "session_id", "datetime",
        "stage_number", "stage_name", "status", "duration_seconds", "result",
        "score", "max_score", "movement", "movement_label", "max_value", "samples",
        "pitch_min", "pitch_max", "yaw_min", "yaw_max", "roll_min", "roll_max"
    ]
    ensure_csv_schema(SESSION_PROGRESS_CSV_PATH, fieldnames)
    file_exists = os.path.exists(SESSION_PROGRESS_CSV_PATH) and os.path.getsize(SESSION_PROGRESS_CSV_PATH) > 0
    profile_id = "guest"
    profile_name = "Guest"
    if current_profile is not None:
        profile_id = current_profile.get("profile_id", "guest")
        profile_name = current_profile.get("name", "Guest")
    snapshot = get_stage_runtime_snapshot(stage_number)
    completed = bool(snapshot.get("completed", False) or str(result).lower() in {"completed", "win", "won"})
    status = "Completed" if completed else "Incomplete"
    duration = get_current_session_elapsed()
    pose = current_session.get("pose", session_empty_pose())
    session_id = current_session.get("session_id", "")
    session_datetime = current_session.get("start_datetime", "")
    with open(SESSION_PROGRESS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for movement_key, movement_data in current_session.get("movements", {}).items():
            writer.writerow({
                "profile_id": profile_id,
                "profile_name": profile_name,
                "session_id": session_id,
                "datetime": session_datetime,
                "stage_number": snapshot.get("stage_number", ""),
                "stage_name": snapshot.get("stage_name", ""),
                "status": status,
                "duration_seconds": f"{duration:.1f}",
                "result": result,
                "score": snapshot.get("score", 0),
                "max_score": snapshot.get("max_score", 0),
                "movement": movement_key,
                "movement_label": movement_data.get("label", movement_key),
                "max_value": f"{movement_data.get('max_value', 0.0):.3f}",
                "samples": movement_data.get("samples", 0),
                "pitch_min": "" if pose.get("pitch_min") is None else f"{pose.get('pitch_min'):.3f}",
                "pitch_max": "" if pose.get("pitch_max") is None else f"{pose.get('pitch_max'):.3f}",
                "yaw_min": "" if pose.get("yaw_min") is None else f"{pose.get('yaw_min'):.3f}",
                "yaw_max": "" if pose.get("yaw_max") is None else f"{pose.get('yaw_max'):.3f}",
                "roll_min": "" if pose.get("roll_min") is None else f"{pose.get('roll_min'):.3f}",
                "roll_max": "" if pose.get("roll_max") is None else f"{pose.get('roll_max'):.3f}",
            })
    save_session_angle_samples(profile_id, profile_name, session_id, session_datetime)
    current_session["saved"] = True
    current_session["active"] = False
    session_saved = True


def load_progress_rows_for_active_profile():
    if not os.path.exists(SESSION_PROGRESS_CSV_PATH):
        return []

    active_id = current_profile.get("profile_id") if current_profile is not None else "guest"
    rows = []
    try:
        with open(SESSION_PROGRESS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("profile_id", "guest") == active_id:
                    rows.append(row)
    except Exception:
        return []
    return rows


def load_analysis_sample_rows_for_active_profile():
    if not os.path.exists(SESSION_ANALYSIS_CSV_PATH):
        return []

    active_id = current_profile.get("profile_id") if current_profile is not None else "guest"
    rows = []
    try:
        with open(SESSION_ANALYSIS_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("profile_id", "guest") == active_id:
                    rows.append(row)
    except Exception:
        return []
    return rows


def safe_float(value, default=0.0):
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except Exception:
        return default


def format_session_datetime_label(value):
    if value is None or value == "":
        return "Unknown session"
    try:
        dt = datetime.fromisoformat(value)
        return dt.strftime("%b %d, %Y - %H:%M")
    except Exception:
        text_value = str(value)
        return text_value[:16]



def build_analysis_sessions(progress_rows):
    sessions = []
    seen = set()
    for row in progress_rows:
        session_id = row.get("session_id", "") or row.get("datetime", "")
        if session_id == "" or session_id in seen:
            continue
        seen.add(session_id)
        sessions.append({
            "session_id": session_id,
            "datetime": row.get("datetime", ""),
            "result": row.get("result", ""),
            "status": row.get("status", ""),
            "score": row.get("score", ""),
            "max_score": row.get("max_score", ""),
            "stage_number": row.get("stage_number", ""),
            "stage_name": row.get("stage_name", ""),
        })
    sessions.sort(key=lambda item: item.get("datetime", ""))
    return sessions


def get_selected_analysis_session(progress_rows):
    global progress_selected_session_index

    sessions = build_analysis_sessions(progress_rows)
    if len(sessions) == 0:
        progress_selected_session_index = None
        return None, sessions, None

    if progress_selected_session_index is None:
        progress_selected_session_index = len(sessions) - 1

    progress_selected_session_index = max(0, min(len(sessions) - 1, int(progress_selected_session_index)))
    return sessions[progress_selected_session_index], sessions, progress_selected_session_index


def get_rows_for_session(progress_rows, session_id):
    if session_id is None:
        return []
    return [row for row in progress_rows if (row.get("session_id", "") or row.get("datetime", "")) == session_id]


def get_analysis_samples_for_session(sample_rows, session_id, movement_key):
    if session_id is None:
        return []
    selected = []
    for row in sample_rows:
        row_session = row.get("session_id", "") or row.get("datetime", "")
        if row_session == session_id and row.get("movement") == movement_key:
            selected.append(row)
    selected.sort(key=lambda item: safe_float(item.get("time_seconds"), 0.0))
    return selected


def estimate_analysis_sample_durations(active_samples):
    """
    Estimate how much active training time each saved sample represents.
    The game writes samples at a fixed interval, but using timestamps makes the
    Analysis page more honest when the frame rate briefly changes.
    """
    if len(active_samples) == 0:
        return []

    times = [safe_float(row.get("time_seconds"), 0.0) for row in active_samples]
    durations = []

    for i, t in enumerate(times):
        if len(times) == 1:
            dt = ANALYSIS_ACTIVE_SAMPLE_TIME_FALLBACK
        elif i < len(times) - 1:
            dt = times[i + 1] - t
        else:
            dt = times[i] - times[i - 1]

        # Avoid counting pauses or duplicate timestamps as huge/zero training time.
        dt = max(0.08, min(0.60, float(dt)))
        durations.append(dt)

    return durations


def get_analysis_quality_score_for_sample(row, movement_key):
    """
    Convert raw calibration distance into a user-facing quality score.
    This intentionally avoids showing harsh debug numbers such as 400x.
    Good active effort is never converted to a discouraging zero; instead,
    the dashboard reports Good / Almost there / Needs control based on time.
    """
    error_value = max(0.0, safe_float(row.get("error_degree"), 0.0))
    metric_value = max(0.0, safe_float(row.get("angle"), 0.0))
    is_correct = int(safe_float(row.get("is_correct"), 0)) == 1

    if movement_key in ANALYSIS_ROM_MOVEMENT_KEYS:
        # ROM error is measured in degrees of unwanted posture drift.
        # 0-8 deg is good, 8-18 deg is still usable but needs smoother control.
        error_quality = 100.0 - min(75.0, error_value * 3.6)
        score = error_quality
    else:
        # Non-ROM error values come from normalized match metrics and can be
        # naturally larger than 100. Use a soft curve plus progress closeness.
        if movement_key == "chin_tuck":
            soft_scale = 520.0
        elif movement_key == "shoulder_lift":
            soft_scale = 620.0
        elif movement_key == "palm_retraction":
            soft_scale = 520.0
        else:
            soft_scale = 560.0

        error_quality = 100.0 / (1.0 + (error_value / max(soft_scale, 1.0)) ** 1.10)

        # metric_value is progress percent for non-ROM movements. A user can be
        # slightly under or over the calibrated target and still be acceptable.
        progress_distance = abs(metric_value - 100.0)
        progress_quality = 100.0 - min(50.0, progress_distance * 0.20)

        score = 0.58 * error_quality + 0.42 * progress_quality

    if is_correct:
        score = max(score, 82.0)

    # If the movement was active, do not collapse the score to a demoralizing 0.
    score = max(35.0, min(100.0, score))
    return score


def summarize_analysis_samples(samples, movement_key=None):
    active = [row for row in samples if int(safe_float(row.get("is_active"), 0)) == 1]

    if len(active) == 0:
        return {
            "active_count": 0,
            "active_time": 0.0,
            "correct_percent": None,
            "quality_percent": None,
            "good_percent": 0.0,
            "acceptable_percent": 0.0,
            "needs_percent": 0.0,
            "good_time": 0.0,
            "acceptable_time": 0.0,
            "needs_time": 0.0,
            "error_percent": None,
            "avg_error": None,
            "avg_quality": None,
            "max_angle": max([safe_float(row.get("angle"), 0.0) for row in samples], default=0.0),
            "consistency": "not enough data",
        }

    durations = estimate_analysis_sample_durations(active)
    if len(durations) != len(active):
        durations = [ANALYSIS_ACTIVE_SAMPLE_TIME_FALLBACK for _ in active]

    active_time = sum(durations)
    if active_time < ANALYSIS_MIN_ACTIVE_TIME_SECONDS:
        return {
            "active_count": len(active),
            "active_time": active_time,
            "correct_percent": None,
            "quality_percent": None,
            "good_percent": 0.0,
            "acceptable_percent": 0.0,
            "needs_percent": 0.0,
            "good_time": 0.0,
            "acceptable_time": 0.0,
            "needs_time": 0.0,
            "error_percent": None,
            "avg_error": None,
            "avg_quality": None,
            "max_angle": max([safe_float(row.get("angle"), 0.0) for row in active], default=0.0),
            "consistency": "not enough active time",
        }

    qualities = [get_analysis_quality_score_for_sample(row, movement_key) for row in active]
    errors = [safe_float(row.get("error_degree"), 0.0) for row in active]
    angles = [safe_float(row.get("angle"), 0.0) for row in active]

    good_time = 0.0
    acceptable_time = 0.0
    needs_time = 0.0

    for quality, dt in zip(qualities, durations):
        if quality >= ANALYSIS_GOOD_QUALITY_PERCENT:
            good_time += dt
        elif quality >= ANALYSIS_ACCEPTABLE_QUALITY_PERCENT:
            acceptable_time += dt
        else:
            needs_time += dt

    weighted_quality = sum(q * dt for q, dt in zip(qualities, durations)) / max(active_time, 0.001)
    good_percent = 100.0 * good_time / max(active_time, 0.001)
    acceptable_percent = 100.0 * acceptable_time / max(active_time, 0.001)
    needs_percent = 100.0 * needs_time / max(active_time, 0.001)

    # Keep old key names for compatibility, but make them mean supportive
    # time-weighted quality rather than a strict pass/fail count.
    correct_percent = weighted_quality
    error_percent = 100.0 - weighted_quality

    avg_error = sum(errors) / max(1, len(errors))

    if len(angles) >= 3:
        avg_angle = sum(angles) / len(angles)
        variance = sum((value - avg_angle) ** 2 for value in angles) / len(angles)
        std_angle = math.sqrt(max(0.0, variance))
        consistency_ratio = std_angle / max(abs(avg_angle), 1.0)
        if consistency_ratio <= 0.28:
            consistency = "good"
        elif consistency_ratio <= 0.55:
            consistency = "medium"
        else:
            consistency = "variable"
    else:
        consistency = "not enough data"

    return {
        "active_count": len(active),
        "active_time": active_time,
        "correct_percent": correct_percent,
        "quality_percent": weighted_quality,
        "good_percent": good_percent,
        "acceptable_percent": acceptable_percent,
        "needs_percent": needs_percent,
        "good_time": good_time,
        "acceptable_time": acceptable_time,
        "needs_time": needs_time,
        "error_percent": error_percent,
        "avg_error": avg_error,
        "avg_quality": weighted_quality,
        "max_angle": max(angles) if len(angles) > 0 else 0.0,
        "consistency": consistency,
    }

def get_analysis_movement_label(movement_key):
    return MOVEMENT_DISPLAY_NAMES.get(movement_key, movement_key)


def count_unique_sessions(rows):
    seen = set()
    for row in rows:
        sid = row.get("session_id", "") or row.get("datetime", "")
        if sid != "":
            seen.add(sid)
    return len(seen)


def count_unique_sessions_for_movement(progress_rows, movement_key):
    return count_unique_sessions([row for row in progress_rows if row.get("movement") == movement_key])


def draw_small_leaf_icon(frame, cx, cy, color=(70, 150, 55)):
    cv2.ellipse(frame, (int(cx - 7), int(cy)), (10, 5), -35, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (int(cx + 7), int(cy - 2)), (10, 5), 35, 0, 360, color, -1, cv2.LINE_AA)
    cv2.line(frame, (int(cx), int(cy + 1)), (int(cx), int(cy + 12)), (65, 120, 55), 2, cv2.LINE_AA)


def draw_tiny_flower_icon(frame, cx, cy, size=18):
    """Small clean Bloom icon for compact controls."""
    cx = int(cx)
    cy = int(cy)
    petal_color = (155, 180, 255)
    petal_border = (110, 125, 220)
    center_color = (245, 250, 255)
    stem_color = (70, 155, 75)

    for angle in range(0, 360, 72):
        rad = math.radians(angle)
        px = int(cx + math.cos(rad) * size * 0.48)
        py = int(cy + math.sin(rad) * size * 0.40)
        cv2.circle(frame, (px, py), max(5, int(size * 0.33)), petal_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), max(5, int(size * 0.33)), petal_border, 1, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), max(6, int(size * 0.36)), center_color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx - 3, cy - 2), 1, (40, 55, 55), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + 3, cy - 2), 1, (40, 55, 55), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy + 2), (4, 3), 0, 0, 180, (40, 55, 55), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + int(size * 0.50)), (cx, cy + int(size * 1.05)), stem_color, 2, cv2.LINE_AA)
    cv2.ellipse(frame, (cx - 6, cy + int(size * 0.88)), (7, 3), -25, 0, 360, stem_color, -1, cv2.LINE_AA)
    cv2.ellipse(frame, (cx + 6, cy + int(size * 0.88)), (7, 3), 25, 0, 360, stem_color, -1, cv2.LINE_AA)


def fit_text_scale(text, max_width, base_scale=0.50, min_scale=0.25, thickness=1):
    """Return a cv2 font scale that keeps text inside max_width."""
    text = str(text)
    scale = float(base_scale)
    while scale > min_scale:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if size[0] <= max_width:
            break
        scale -= 0.025
    return max(min_scale, scale)


def ellipsize_text_to_width(text, max_width, scale=0.45, thickness=1):
    text = str(text)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if size[0] <= max_width:
        return text

    while len(text) > 4:
        candidate = text[:-4] + "..."
        size, _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if size[0] <= max_width:
            return candidate
        text = text[:-1]

    return "..."


def draw_ui_text_fit(frame, text, x, y, max_width, base_scale=0.50, min_scale=0.28, color=(40, 55, 65), thickness=1):
    text = str(text)
    scale = fit_text_scale(text, max_width, base_scale=base_scale, min_scale=min_scale, thickness=thickness)
    if scale <= min_scale + 0.001:
        text = ellipsize_text_to_width(text, max_width, scale=scale, thickness=thickness)
    draw_ui_text(frame, text, x, y, scale=scale, color=color, thickness=thickness)


def draw_centered_text_fit(frame, text, center_x, y, max_width, base_scale=0.50, min_scale=0.28, color=(40, 55, 65), thickness=1):
    text = str(text)
    scale = fit_text_scale(text, max_width, base_scale=base_scale, min_scale=min_scale, thickness=thickness)
    if scale <= min_scale + 0.001:
        text = ellipsize_text_to_width(text, max_width, scale=scale, thickness=thickness)
    draw_centered_text(frame, text, center_x, y, scale=scale, color=color, thickness=thickness)

def draw_analysis_page_frame():
    """Clean, spacious Analysis page base with no bottom feedback strip."""
    frame = draw_menu_background(overlay_alpha=0.06 if main_menu_background_img is not None else 0.18)

    draw_transparent_rounded_rect(frame, 84, 64, 1196, 690, (255, 252, 235), alpha=0.90, radius=42)
    draw_rounded_rect(frame, 84, 64, 1196, 690, (95, 155, 85), radius=42, thickness=5)

    draw_centered_text(frame, "Analysis", WIDTH // 2 + 3, 51 + 3, scale=1.42, color=(255, 255, 255), thickness=7)
    draw_centered_text(frame, "Analysis", WIDTH // 2, 51, scale=1.42, color=(45, 125, 55), thickness=5)
    draw_small_leaf_icon(frame, WIDTH // 2 - 155, 42, color=(75, 150, 70))
    draw_small_leaf_icon(frame, WIDTH // 2 + 155, 42, color=(75, 150, 70))

    return frame

def get_progress_buttons():
    """Buttons for the Analysis screen."""
    buttons = [
        {"id": "movement_dropdown", "label": "Movements", "rect": (120, 88, 430, 148)},
        {"id": "session_prev", "label": "< Previous", "rect": (820, 88, 975, 148)},
        {"id": "session_next", "label": "Next >", "rect": (995, 88, 1150, 148)},
        {"id": "back_main", "label": "Back", "rect": (1000, 623, 1172, 678)},
    ]

    if progress_movement_dropdown_open:
        option_x1, option_x2 = 120, 430
        option_y = 154
        option_h = 36
        for i, (mid, label) in enumerate(ANALYSIS_ALL_MOVEMENTS):
            buttons.append({
                "id": f"progress_{mid}",
                "label": label,
                "rect": (option_x1, option_y + i * option_h, option_x2, option_y + (i + 1) * option_h - 3)
            })

    return buttons

def draw_movement_dropdown_control(frame, buttons, hover_button):
    selected_label = get_analysis_movement_label(progress_selected_movement)
    main_button = next((b for b in buttons if b["id"] == "movement_dropdown"), None)
    if main_button is None:
        return

    x1, y1, x2, y2 = main_button["rect"]
    hovered = hover_button == "movement_dropdown"
    fill = (255, 255, 250) if not hovered else (250, 255, 238)
    border = (120, 165, 95) if not progress_movement_dropdown_open else (50, 150, 75)

    draw_transparent_rounded_rect(frame, x1 + 4, y1 + 5, x2 + 4, y2 + 5, (55, 75, 55), alpha=0.15, radius=20)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, fill, radius=20)
    draw_rounded_rect(frame, x1, y1, x2, y2, border, radius=20, thickness=3)

    icon_x = x1 + 34
    icon_y = y1 + 28
    cv2.circle(frame, (icon_x, icon_y), 25, (238, 250, 238), -1, cv2.LINE_AA)
    cv2.circle(frame, (icon_x, icon_y), 25, (190, 220, 185), 1, cv2.LINE_AA)
    draw_tiny_flower_icon(frame, icon_x, icon_y - 3, size=18)

    draw_ui_text(frame, "Movements", x1 + 70, y1 + 21, scale=0.40, color=(70, 80, 75), thickness=1)
    draw_ui_text_fit(frame, selected_label, x1 + 70, y1 + 48, x2 - x1 - 122, base_scale=0.54, min_scale=0.32, color=(25, 45, 40), thickness=2)

    if progress_selected_movement in ANALYSIS_ROM_MOVEMENT_KEYS:
        badge_text = "Angle"
        badge_fill = (225, 248, 222)
        badge_border = (85, 175, 80)
        badge_color = (45, 130, 55)
    else:
        badge_text = "Quality"
        badge_fill = (238, 248, 255)
        badge_border = (115, 170, 205)
        badge_color = (55, 105, 145)

    badge_x1, badge_y1 = x2 - 94, y1 + 10
    badge_x2, badge_y2 = x2 - 36, y1 + 34
    draw_filled_rounded_rect(frame, badge_x1, badge_y1, badge_x2, badge_y2, badge_fill, radius=10)
    draw_rounded_rect(frame, badge_x1, badge_y1, badge_x2, badge_y2, badge_border, radius=10, thickness=1)
    draw_centered_text_fit(frame, badge_text, int((badge_x1 + badge_x2) / 2), badge_y1 + 16, badge_x2 - badge_x1 - 6, base_scale=0.32, min_scale=0.25, color=badge_color, thickness=1)

    chevron_x = x2 - 20
    chevron_y = y1 + 33
    if progress_movement_dropdown_open:
        pts = np.array([[chevron_x - 7, chevron_y + 4], [chevron_x, chevron_y - 4], [chevron_x + 7, chevron_y + 4]], dtype=np.int32)
    else:
        pts = np.array([[chevron_x - 7, chevron_y - 4], [chevron_x, chevron_y + 4], [chevron_x + 7, chevron_y - 4]], dtype=np.int32)
    cv2.polylines(frame, [pts], False, (35, 110, 55), 3, cv2.LINE_AA)

    if not progress_movement_dropdown_open:
        return

    options = [b for b in buttons if b["id"].startswith("progress_")]
    if len(options) == 0:
        return

    list_x1 = x1
    list_y1 = options[0]["rect"][1] - 5
    list_x2 = x2
    list_y2 = options[-1]["rect"][3] + 6
    draw_transparent_rounded_rect(frame, list_x1 + 5, list_y1 + 6, list_x2 + 5, list_y2 + 6, (55, 75, 55), alpha=0.18, radius=18)
    draw_filled_rounded_rect(frame, list_x1, list_y1, list_x2, list_y2, (255, 255, 250), radius=18)
    draw_rounded_rect(frame, list_x1, list_y1, list_x2, list_y2, (115, 160, 95), radius=18, thickness=2)

    for option in options:
        ox1, oy1, ox2, oy2 = option["rect"]
        mid = option["id"].replace("progress_", "")
        selected = mid == progress_selected_movement
        hovered_option = hover_button == option["id"]

        if selected or hovered_option:
            fill = (229, 250, 222) if selected else (244, 252, 236)
            draw_filled_rounded_rect(frame, ox1 + 8, oy1 + 3, ox2 - 8, oy2 - 3, fill, radius=11)

        if selected:
            cv2.circle(frame, (ox1 + 22, int((oy1 + oy2) / 2)), 5, (60, 160, 75), -1, cv2.LINE_AA)

        draw_ui_text_fit(frame, option["label"], ox1 + 40, oy1 + 23, 176, base_scale=0.40, min_scale=0.30, color=(35, 65, 45), thickness=1)

        tag = "angle" if mid in ANALYSIS_ROM_MOVEMENT_KEYS else "quality"
        tag_color = (55, 135, 65) if mid in ANALYSIS_ROM_MOVEMENT_KEYS else (55, 105, 145)
        draw_ui_text_fit(frame, tag, ox2 - 72, oy1 + 23, 56, base_scale=0.30, min_scale=0.23, color=tag_color, thickness=1)


def draw_analysis_session_card(frame, session, sessions, selected_index):
    x1, y1, x2, y2 = 455, 88, 800, 148
    draw_transparent_rounded_rect(frame, x1 + 4, y1 + 5, x2 + 4, y2 + 5, (55, 75, 55), alpha=0.14, radius=20)
    draw_filled_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 250), radius=20)
    draw_rounded_rect(frame, x1, y1, x2, y2, (140, 175, 110), radius=20, thickness=2)
    cal_x, cal_y = x1 + 20, y1 + 15
    cv2.rectangle(frame, (cal_x, cal_y + 7), (cal_x + 34, cal_y + 36), (245, 255, 245), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (cal_x, cal_y + 7), (cal_x + 34, cal_y + 36), (55, 130, 70), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (cal_x, cal_y + 7), (cal_x + 34, cal_y + 16), (80, 170, 75), -1, cv2.LINE_AA)
    if session is None:
        session_text = "No saved session"
        detail_text = "Play any stage first"
    else:
        total = len(sessions)
        current = selected_index + 1 if selected_index is not None else total
        stage_number = session.get("stage_number", "")
        session_text = f"Session {current}/{total}" + (f" - S{stage_number}" if stage_number else "")
        stage_name = session.get("stage_name", "")
        status = session.get("status", "")
        score_value = session.get("score", "")
        max_score = session.get("max_score", "")
        score_text = f"{score_value}/{max_score}" if max_score not in ["", None, "0", 0] else str(score_value)
        parts = [part for part in [stage_name, status, score_text, format_session_datetime_label(session.get("datetime", ""))] if part]
        detail_text = " | ".join(parts)
    draw_ui_text_fit(frame, session_text, x1 + 68, y1 + 35, 135, base_scale=0.50, min_scale=0.31, color=(25, 45, 40), thickness=2)
    cv2.line(frame, (x1 + 205, y1 + 14), (x1 + 205, y2 - 14), (200, 215, 195), 1, cv2.LINE_AA)
    draw_ui_text_fit(frame, detail_text, x1 + 218, y1 + 35, x2 - x1 - 230, base_scale=0.36, min_scale=0.25, color=(45, 65, 60), thickness=1)

def draw_axis_line_chart(frame, values, x1, y1, x2, y2, y_label="Angle", empty_title=None, empty_subtitle=None):
    plot_x1 = x1 + 66
    plot_y1 = y1 + 76
    plot_x2 = x2 - 34
    plot_y2 = y2 - 54

    draw_filled_rounded_rect(frame, plot_x1 - 18, plot_y1 - 24, plot_x2 + 16, plot_y2 + 36, (252, 254, 252), radius=18)

    cv2.line(frame, (plot_x1, plot_y2), (plot_x2, plot_y2), (120, 130, 135), 2, cv2.LINE_AA)
    cv2.line(frame, (plot_x1, plot_y1), (plot_x1, plot_y2), (120, 130, 135), 2, cv2.LINE_AA)
    draw_ui_text(frame, y_label, x1 + 24, y1 + 70, scale=0.35, color=(55, 70, 75), thickness=1)

    if len(values) == 0:
        if empty_title is None:
            empty_title = "No angle-time samples for this session yet"
        if empty_subtitle is None:
            empty_subtitle = "Play a new session with this version to generate the chart"

        cx = int((plot_x1 + plot_x2) / 2)
        cy = int((plot_y1 + plot_y2) / 2)
        cv2.circle(frame, (cx, cy - 38), 20, (235, 245, 238), -1, cv2.LINE_AA)
        draw_small_leaf_icon(frame, cx, cy - 48, color=(110, 170, 100))
        draw_centered_text_fit(frame, empty_title, cx, cy + 2, plot_x2 - plot_x1 - 40, base_scale=0.43, min_scale=0.30, color=(80, 95, 105), thickness=1)
        draw_centered_text_fit(frame, empty_subtitle, cx, cy + 30, plot_x2 - plot_x1 - 40, base_scale=0.34, min_scale=0.26, color=(95, 105, 115), thickness=1)
        return

    times = [safe_float(row.get("time_seconds"), 0.0) for row in values]
    angles = [safe_float(row.get("angle"), 0.0) for row in values]

    min_time = min(times) if len(times) > 0 else 0.0
    max_time = max(times) if len(times) > 0 else 1.0
    if max_time <= min_time:
        max_time = min_time + 1.0

    max_angle = max(max(angles), 10.0)
    max_angle = max(15.0, max_angle * 1.18)

    for i in range(5):
        yy = int(plot_y2 - i * (plot_y2 - plot_y1) / 4)
        cv2.line(frame, (plot_x1, yy), (plot_x2, yy), (226, 232, 232), 1, cv2.LINE_AA)
        tick = max_angle * i / 4
        draw_ui_text(frame, f"{tick:.0f}", plot_x1 - 46, yy + 5, scale=0.33, color=(80, 92, 98), thickness=1)

    for i in range(5):
        xx = int(plot_x1 + i * (plot_x2 - plot_x1) / 4)
        cv2.line(frame, (xx, plot_y2), (xx, plot_y2 + 5), (140, 150, 150), 1, cv2.LINE_AA)
        tick_time = min_time + (max_time - min_time) * i / 4
        draw_centered_text(frame, f"{tick_time:.0f}", xx, plot_y2 + 23, scale=0.30, color=(80, 92, 98), thickness=1)

    points = []
    for t, a in zip(times, angles):
        px = int(plot_x1 + ((t - min_time) / (max_time - min_time)) * (plot_x2 - plot_x1))
        py = int(plot_y2 - (a / max_angle) * (plot_y2 - plot_y1))
        points.append((px, py))

    line_color = (235, 120, 20)
    fill_color = (245, 250, 255)

    if len(points) >= 2:
        fill_poly = np.array(points + [(points[-1][0], plot_y2), (points[0][0], plot_y2)], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [fill_poly], fill_color, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], line_color, 3, cv2.LINE_AA)

    max_angle_value = max(angles) if len(angles) > 0 else 0.0
    for i, (px, py) in enumerate(points):
        if i == 0 or i == len(points) - 1 or angles[i] == max_angle_value:
            cv2.circle(frame, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, line_color, -1, cv2.LINE_AA)

    draw_centered_text(frame, "Time (s)", int((plot_x1 + plot_x2) / 2), y2 - 16, scale=0.34, color=(55, 70, 75), thickness=1)

def draw_donut_accuracy(frame, x1, y1, x2, y2, summary, movement_key=None):
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), alpha=0.96, radius=26)
    draw_rounded_rect(frame, x1, y1, x2, y2, (125, 170, 100), radius=26, thickness=2)

    draw_small_leaf_icon(frame, x1 + 28, y1 + 25, color=(85, 155, 70))
    draw_ui_text_fit(frame, "Calibration Accuracy", x1 + 50, y1 + 36, x2 - x1 - 72, base_scale=0.48, min_scale=0.34, color=(30, 95, 45), thickness=2)

    cx = x1 + 95
    cy = y1 + 92
    radius = 42
    thickness = 17

    quality_percent = summary.get("quality_percent", summary.get("correct_percent"))
    active_time = float(summary.get("active_time", 0.0) or 0.0)

    if quality_percent is None:
        cv2.circle(frame, (cx, cy), radius, (218, 224, 224), thickness, cv2.LINE_AA)
        draw_centered_text(frame, "N/A", cx, cy + 7, scale=0.55, color=(90, 100, 110), thickness=2)
        if active_time > 0:
            message = f"Active time: {active_time:.1f}s"
            sub = "Need a little more time"
        else:
            message = "No active movement time"
            sub = "Play this movement again"
        draw_ui_text_fit(frame, message, x1 + 185, y1 + 82, x2 - x1 - 205, base_scale=0.36, min_scale=0.27, color=(90, 100, 110), thickness=1)
        draw_ui_text_fit(frame, sub, x1 + 185, y1 + 110, x2 - x1 - 205, base_scale=0.32, min_scale=0.25, color=(100, 110, 116), thickness=1)
        return

    quality_percent = max(0.0, min(100.0, float(quality_percent)))
    good_percent = max(0.0, min(100.0, float(summary.get("good_percent", 0.0))))
    acceptable_percent = max(0.0, min(100.0, float(summary.get("acceptable_percent", 0.0))))
    needs_percent = max(0.0, min(100.0, 100.0 - good_percent - acceptable_percent))

    green = (85, 175, 65)
    amber = (70, 160, 215)
    soft_red = (95, 105, 235)
    track = (226, 232, 226)

    cv2.circle(frame, (cx, cy), radius, track, thickness, cv2.LINE_AA)

    start_angle = -90
    good_angle = int(360 * good_percent / 100.0)
    acceptable_angle = int(360 * acceptable_percent / 100.0)
    needs_angle = max(0, 360 - good_angle - acceptable_angle)

    if good_angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), start_angle, 0, good_angle, green, thickness, cv2.LINE_AA)
    if acceptable_angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), start_angle + good_angle, 0, acceptable_angle, amber, thickness, cv2.LINE_AA)
    if needs_angle > 0:
        cv2.ellipse(frame, (cx, cy), (radius, radius), start_angle + good_angle + acceptable_angle, 0, needs_angle, soft_red, thickness, cv2.LINE_AA)

    cv2.circle(frame, (cx, cy), radius - thickness // 2 - 3, (255, 255, 255), -1, cv2.LINE_AA)
    draw_centered_text(frame, f"{quality_percent:.0f}%", cx, cy + 8, scale=0.64, color=(45, 135, 55), thickness=3)

    legend_x = x1 + 205
    value_x = x2 - 82
    row_y = y1 + 67
    good_time = float(summary.get("good_time", 0.0) or 0.0)
    acceptable_time = float(summary.get("acceptable_time", 0.0) or 0.0)
    needs_time = float(summary.get("needs_time", 0.0) or 0.0)

    legend_rows = [
        ("Good", f"{good_time:.1f}s", green),
        ("Almost", f"{acceptable_time:.1f}s", amber),
        ("Needs", f"{needs_time:.1f}s", soft_red),
        ("Active", f"{active_time:.1f}s", (80, 135, 190)),
    ]

    for label, value, color in legend_rows:
        cv2.circle(frame, (legend_x, row_y), 7, color, -1, cv2.LINE_AA)
        draw_ui_text_fit(frame, label, legend_x + 20, row_y + 5, 82, base_scale=0.34, min_scale=0.25, color=(45, 60, 55), thickness=1)
        draw_ui_text_fit(frame, value, value_x, row_y + 5, 66, base_scale=0.34, min_scale=0.25, color=(35, 45, 40), thickness=2)
        row_y += 25

def get_analysis_error_detail_rows(summary, movement_key):
    quality = summary.get("quality_percent", summary.get("correct_percent"))
    active_time = float(summary.get("active_time", 0.0) or 0.0)
    active_count = int(summary.get("active_count", 0))
    consistency = summary.get("consistency", "not enough data")

    if movement_key in ANALYSIS_ROM_MOVEMENT_KEYS:
        quality_label = "Angle quality"
        control_label = "Posture control"
        good_suggestion = "Keep smooth"
        medium_suggestion = "Move slower"
        hard_suggestion = "Reduce rotation"
    elif movement_key == "chin_tuck":
        quality_label = "Chin control"
        control_label = "Head stability"
        good_suggestion = "Keep steady"
        medium_suggestion = "Face forward"
        hard_suggestion = "Try smaller motion"
    elif movement_key == "shoulder_lift":
        quality_label = "Shoulder quality"
        control_label = "Shoulder balance"
        good_suggestion = "Lift evenly"
        medium_suggestion = "Balance shoulders"
        hard_suggestion = "Relax and repeat"
    elif movement_key == "palm_retraction":
        quality_label = "Palm quality"
        control_label = "Palm balance"
        good_suggestion = "Keep palms even"
        medium_suggestion = "Move both palms out"
        hard_suggestion = "Control both sides"
    else:
        quality_label = "Movement quality"
        control_label = "Movement control"
        good_suggestion = "Keep smooth"
        medium_suggestion = "Control motion"
        hard_suggestion = "Repeat gently"

    if quality is None:
        if active_time > 0:
            return [
                (quality_label, "Need more time", (95, 100, 110)),
                ("Active time", f"{active_time:.1f}s", (70, 135, 210)),
                (control_label, "Not rated", (95, 100, 110)),
                ("Suggestion", "Try 2s more", (65, 120, 80)),
            ]
        return [
            (quality_label, "No data", (95, 100, 110)),
            ("Active time", "0.0s", (95, 100, 110)),
            (control_label, "N/A", (95, 100, 110)),
            ("Suggestion", "Play movement", (65, 120, 80)),
        ]

    quality = max(0.0, min(100.0, float(quality)))

    if quality >= ANALYSIS_GOOD_QUALITY_PERCENT:
        control_value = "Good"
        control_color = (55, 150, 75)
        suggestion = good_suggestion
    elif quality >= ANALYSIS_ACCEPTABLE_QUALITY_PERCENT:
        control_value = "Almost there"
        control_color = (70, 135, 210)
        suggestion = medium_suggestion
    else:
        control_value = "Needs control"
        control_color = (70, 80, 230)
        suggestion = hard_suggestion

    if consistency == "variable" and quality >= ANALYSIS_GOOD_QUALITY_PERCENT:
        control_value = "Variable"
        control_color = (70, 135, 210)
        suggestion = medium_suggestion

    sample_color = (55, 150, 75) if active_count >= 10 else (70, 135, 210)

    return [
        (quality_label, f"{quality:.0f}%", (45, 130, 65)),
        ("Active time", f"{active_time:.1f}s", sample_color),
        (control_label, control_value, control_color),
        ("Suggestion", suggestion, (45, 130, 65)),
    ]

def draw_error_details_card(frame, x1, y1, x2, y2, summary, movement_key=None):
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), alpha=0.96, radius=26)
    draw_rounded_rect(frame, x1, y1, x2, y2, (125, 170, 100), radius=26, thickness=2)

    draw_small_leaf_icon(frame, x1 + 28, y1 + 25, color=(85, 155, 70))
    draw_ui_text(frame, "Error Details", x1 + 50, y1 + 35, scale=0.48, color=(30, 95, 45), thickness=2)

    rows = get_analysis_error_detail_rows(summary, movement_key)

    row_h = 27
    row_box_h = 23
    row_y = y1 + 49
    icon_colors = [(45, 135, 235), (75, 160, 85), (80, 135, 190), (185, 100, 220)]

    for i, (label, value, value_color) in enumerate(rows[:4]):
        ry1 = row_y + i * row_h
        ry2 = ry1 + row_box_h
        if ry2 > y2 - 8:
            break
        draw_filled_rounded_rect(frame, x1 + 16, ry1, x2 - 16, ry2, (248, 252, 248), radius=11)
        draw_rounded_rect(frame, x1 + 16, ry1, x2 - 16, ry2, (220, 230, 218), radius=11, thickness=1)
        cv2.circle(frame, (x1 + 35, ry1 + 12), 7, icon_colors[i], 1, cv2.LINE_AA)
        draw_ui_text_fit(frame, label, x1 + 52, ry1 + 16, 168, base_scale=0.31, min_scale=0.24, color=(50, 65, 60), thickness=1)

        value_text = str(value)
        value_max_width = 112
        value_scale = fit_text_scale(value_text, value_max_width, base_scale=0.31, min_scale=0.23, thickness=1)
        value_text = ellipsize_text_to_width(value_text, value_max_width, scale=value_scale, thickness=1)
        value_size, _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, value_scale, 1)
        draw_ui_text(frame, value_text, x2 - 24 - value_size[0], ry1 + 16, scale=value_scale, color=value_color, thickness=1)

def build_quality_trend_rows_from_samples(sample_rows, movement_key):
    """Build per-session quality trend rows for non-ROM movements."""
    grouped = {}
    session_datetime = {}

    for row in sample_rows:
        if row.get("movement") != movement_key:
            continue
        sid = row.get("session_id", "") or row.get("datetime", "")
        if sid == "":
            continue
        grouped.setdefault(sid, []).append(row)
        session_datetime[sid] = row.get("datetime", "")

    rows = []
    for sid, rows_for_session in grouped.items():
        summary = summarize_analysis_samples(rows_for_session, movement_key)
        quality = summary.get("quality_percent")
        if quality is None:
            continue
        rows.append({
            "session_id": sid,
            "datetime": session_datetime.get(sid, ""),
            "max_value": f"{float(quality):.3f}",
            "movement": movement_key,
        })

    rows.sort(key=lambda item: item.get("datetime", ""))
    return rows


def draw_progress_trend_compact(frame, progress_rows, movement_key, x1, y1, x2, y2, sample_rows=None):
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 255), alpha=0.96, radius=26)
    draw_rounded_rect(frame, x1, y1, x2, y2, (105, 165, 80), radius=26, thickness=2)

    draw_small_leaf_icon(frame, x1 + 28, y1 + 28, color=(85, 155, 70))
    draw_ui_text(frame, "Progress Trend", x1 + 50, y1 + 37, scale=0.54, color=(30, 95, 45), thickness=2)

    use_quality_trend = movement_key not in ANALYSIS_ROM_MOVEMENT_KEYS and sample_rows is not None
    if use_quality_trend:
        movement_rows_all = build_quality_trend_rows_from_samples(sample_rows, movement_key)
        y_axis_label = "Quality"
        y_max_fixed = 100.0
    else:
        movement_rows_all = [row for row in progress_rows if row.get("movement") == movement_key]
        y_axis_label = "Value"
        y_max_fixed = None

    movement_rows = movement_rows_all[-20:]
    movement_session_count = count_unique_sessions_for_movement(progress_rows, movement_key)
    all_session_count = count_unique_sessions(progress_rows)

    chip_y1, chip_y2 = y1 + 17, y1 + 47
    chip_specs = [
        (x2 - 395, x2 - 270, f"This move: {movement_session_count}", (250, 253, 248), (210, 225, 200)),
        (x2 - 260, x2 - 150, f"All: {all_session_count}", (250, 253, 248), (210, 225, 200)),
        (x2 - 140, x2 - 22, "Last 20", (248, 250, 230), (210, 220, 170)),
    ]
    for cx1, cx2, label, fill, border in chip_specs:
        draw_filled_rounded_rect(frame, cx1, chip_y1, cx2, chip_y2, fill, radius=12)
        draw_rounded_rect(frame, cx1, chip_y1, cx2, chip_y2, border, radius=12, thickness=1)
        draw_centered_text_fit(frame, label, int((cx1 + cx2) / 2), chip_y1 + 20, cx2 - cx1 - 12, base_scale=0.32, min_scale=0.25, color=(45, 70, 55), thickness=1)

    plot_x1 = x1 + 76
    plot_y1 = y1 + 69
    plot_x2 = x2 - 42
    plot_y2 = y2 - 46

    cv2.line(frame, (plot_x1, plot_y2), (plot_x2, plot_y2), (120, 130, 135), 1, cv2.LINE_AA)
    cv2.line(frame, (plot_x1, plot_y1), (plot_x1, plot_y2), (120, 130, 135), 1, cv2.LINE_AA)

    draw_ui_text(frame, y_axis_label, x1 + 21, plot_y1 + 4, scale=0.31, color=(65, 75, 80), thickness=1)

    if len(movement_rows) == 0:
        empty_msg = "No quality history for this movement yet" if use_quality_trend else "No progress history for this movement yet"
        draw_centered_text_fit(frame, empty_msg, int((plot_x1 + plot_x2) / 2), int((plot_y1 + plot_y2) / 2), plot_x2 - plot_x1 - 40, base_scale=0.40, min_scale=0.28, color=(90, 100, 105), thickness=1)
        return

    values = [safe_float(row.get("max_value"), 0.0) for row in movement_rows]
    if y_max_fixed is not None:
        max_value = y_max_fixed
    else:
        max_value = max(max(values), 10.0) * 1.18

    for i in range(5):
        yy = int(plot_y2 - i * (plot_y2 - plot_y1) / 4)
        cv2.line(frame, (plot_x1, yy), (plot_x2, yy), (228, 235, 235), 1, cv2.LINE_AA)
        tick = max_value * i / 4
        draw_ui_text(frame, f"{tick:.0f}", plot_x1 - 45, yy + 5, scale=0.30, color=(80, 92, 98), thickness=1)

    total_rows = len(movement_rows_all)
    start_session_num = max(1, total_rows - len(movement_rows) + 1)

    points = []
    n = len(values)
    green = (65, 150, 60)
    for i, value in enumerate(values):
        value = max(0.0, min(max_value, value))
        px = int((plot_x1 + plot_x2) / 2) if n == 1 else int(plot_x1 + i * (plot_x2 - plot_x1) / (n - 1))
        py = int(plot_y2 - (value / max_value) * (plot_y2 - plot_y1))
        points.append((px, py))

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], green, 3, cv2.LINE_AA)

    label_every = 1 if len(points) <= 12 else 2
    for i, (px, py) in enumerate(points):
        cv2.circle(frame, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 4, green, -1, cv2.LINE_AA)
        if i % label_every == 0 or i == len(points) - 1:
            session_number = start_session_num + i
            draw_centered_text(frame, str(session_number), px, plot_y2 + 21, scale=0.27, color=(70, 84, 84), thickness=1)

    max_i = values.index(max(values))
    label_indices = sorted(set([max_i, len(values) - 1]))
    for i in label_indices:
        px, py = points[i]
        suffix = "%" if use_quality_trend else ""
        draw_centered_text(frame, f"{values[i]:.0f}{suffix}", px, py - 10, scale=0.30, color=(45, 95, 50), thickness=1)

    x_label = "Quality history" if use_quality_trend else "Session history"
    draw_centered_text(frame, x_label, int((plot_x1 + plot_x2) / 2), y2 - 14, scale=0.32, color=(55, 70, 75), thickness=1)

def build_analysis_feedback(summary, progress_rows, movement_key):
    movement_name = MOVEMENT_DISPLAY_NAMES.get(movement_key, movement_key)
    movement_rows = [row for row in progress_rows if row.get("movement") == movement_key]
    values = [safe_float(row.get("max_value"), 0.0) for row in movement_rows[-5:]]

    messages = []

    if len(values) >= 2:
        change = values[-1] - values[0]
        if change > 2.0:
            messages.append("You improved compared to previous sessions.")
        elif change < -2.0:
            messages.append("Your ROM is lower than previous sessions; move gently and avoid forcing it.")
        else:
            messages.append("Your ROM is stable compared to recent sessions.")
    else:
        messages.append("New sessions will make the progress trend more meaningful.")

    correct_percent = summary.get("correct_percent")
    avg_error = summary.get("avg_error")
    max_angle = summary.get("max_angle", 0.0)

    if correct_percent is not None:
        if correct_percent >= ANALYSIS_GOOD_ACCURACY_PERCENT:
            messages.append("Good movement quality and calibration control.")
        else:
            messages.append("Try to keep your head straighter during the movement.")
    else:
        messages.append("Do a few clearer repetitions so accuracy can be calculated.")

    if avg_error is not None and avg_error > ANALYSIS_MEDIUM_ERROR_DEG:
        messages.append("Reduce unwanted rotation and stay closer to your calibrated posture.")
    elif max_angle > 0:
        messages.append(f"Best {movement_name} angle in this session: {max_angle:.1f} deg.")

    if movement_key in ["flexion", "extension"]:
        messages.append("Keep your face forward; avoid turning left or right while moving.")
    else:
        messages.append("Bend sideways without extra forward/backward neck movement.")

    return messages[:4]


def draw_blooms_feedback_card(frame, x1, y1, x2, y2, messages):
    """
    Bottom feedback card. This replaces the old Session Summary section so the
    page has more space and the feedback is readable.
    """
    draw_transparent_rounded_rect(frame, x1, y1, x2, y2, (255, 255, 250), alpha=0.96, radius=24)
    draw_rounded_rect(frame, x1, y1, x2, y2, (135, 175, 100), radius=24, thickness=2)

    # Simple Bloom flower mascot.
    face_cx, face_cy = x1 + 62, y1 + 44
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        px = int(face_cx + math.cos(rad) * 25)
        py = int(face_cy + math.sin(rad) * 22)
        cv2.circle(frame, (px, py), 17, (150, 170, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 17, (105, 120, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (face_cx, face_cy), 24, (235, 248, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (face_cx - 8, face_cy - 5), 3, (35, 50, 55), -1, cv2.LINE_AA)
    cv2.circle(frame, (face_cx + 8, face_cy - 5), 3, (35, 50, 55), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (face_cx, face_cy + 5), (9, 6), 0, 0, 180, (35, 50, 55), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (face_cx - 12, face_cy + 34), (14, 6), -25, 0, 360, (75, 165, 75), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (face_cx + 12, face_cy + 34), (14, 6), 25, 0, 360, (75, 165, 75), -1, cv2.LINE_AA)

    draw_small_leaf_icon(frame, x1 + 142, y1 + 28, color=(85, 155, 70))
    draw_ui_text(frame, "Bloom's Feedback", x1 + 162, y1 + 36, scale=0.55, color=(30, 95, 45), thickness=2)

    if len(messages) == 0:
        main_msg = "Play a session to generate feedback."
        cue_msg = "Coaching cue: move gently and stay inside a comfortable range."
    else:
        main_msg = messages[0]
        cue_msg = "Coaching cue: " + (messages[1] if len(messages) > 1 else "move smoothly and keep your posture controlled.")

    # Main message: wrap into at most two lines.
    main_lines = wrap_text_lines(main_msg, x2 - x1 - 290, scale=0.37, thickness=1)
    for i, line in enumerate(main_lines[:2]):
        draw_ui_text(frame, line, x1 + 162, y1 + 63 + i * 18, scale=0.37, color=(35, 50, 50), thickness=1)

    # Coaching cue line.
    cue_lines = wrap_text_lines(cue_msg, x2 - x1 - 300, scale=0.34, thickness=1)
    star_x = x1 + 164
    star_y = y2 - 22
    cv2.circle(frame, (star_x, star_y - 4), 7, (20, 185, 245), -1, cv2.LINE_AA)
    if len(cue_lines) > 0:
        draw_ui_text(frame, cue_lines[0], x1 + 184, y2 - 20, scale=0.34, color=(55, 70, 55), thickness=1)

    # Small watering can / sprout icon on the right.
    icon_cx = x2 - 78
    icon_cy = int((y1 + y2) / 2)
    cv2.circle(frame, (icon_cx, icon_cy), 42, (252, 252, 238), -1, cv2.LINE_AA)
    cv2.circle(frame, (icon_cx, icon_cy), 42, (220, 225, 185), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (icon_cx - 4, icon_cy - 5), (26, 17), -15, 0, 360, (190, 160, 85), 3, cv2.LINE_AA)
    cv2.line(frame, (icon_cx - 26, icon_cy - 2), (icon_cx - 42, icon_cy + 8), (190, 160, 85), 3, cv2.LINE_AA)
    for i in range(3):
        cv2.circle(frame, (icon_cx - 47 - i * 8, icon_cy + 15 + i * 8), 3, (225, 160, 70), -1, cv2.LINE_AA)
    cv2.line(frame, (icon_cx + 6, icon_cy + 24), (icon_cx + 6, icon_cy + 35), (70, 145, 70), 2, cv2.LINE_AA)
    cv2.ellipse(frame, (icon_cx - 3, icon_cy + 30), (10, 5), -25, 0, 360, (75, 165, 75), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (icon_cx + 15, icon_cy + 30), (10, 5), 25, 0, 360, (75, 165, 75), -1, cv2.LINE_AA)


def draw_analysis_header(frame, session, sessions, selected_index):
    profile_name = current_profile.get("name", "Guest") if current_profile is not None else "Guest"
    draw_ui_text_fit(frame, f"Profile: {profile_name}", 124, 79, 300, base_scale=0.34, min_scale=0.25, color=(75, 95, 75), thickness=1)
    draw_analysis_session_card(frame, session, sessions, selected_index)

def draw_progress_screen():
    frame = draw_analysis_page_frame()

    progress_rows = load_progress_rows_for_active_profile()
    sample_rows = load_analysis_sample_rows_for_active_profile()
    selected_session, sessions, selected_index = get_selected_analysis_session(progress_rows)
    selected_session_id = selected_session.get("session_id") if selected_session is not None else None

    if selected_session_id == "":
        selected_session_id = selected_session.get("datetime") if selected_session is not None else None

    draw_analysis_header(frame, selected_session, sessions, selected_index)

    buttons = get_progress_buttons()
    hover_button = get_button_at_position(mouse_x, mouse_y, buttons)

    is_rom_movement = progress_selected_movement in ANALYSIS_ROM_MOVEMENT_KEYS
    samples = get_analysis_samples_for_session(sample_rows, selected_session_id, progress_selected_movement)
    summary = summarize_analysis_samples(samples, progress_selected_movement)

    movement_label = get_analysis_movement_label(progress_selected_movement)

    chart_x1, chart_y1, chart_x2, chart_y2 = 115, 170, 785, 465
    draw_transparent_rounded_rect(frame, chart_x1, chart_y1, chart_x2, chart_y2, (255, 255, 255), alpha=0.96, radius=28)
    draw_rounded_rect(frame, chart_x1, chart_y1, chart_x2, chart_y2, (105, 165, 80), radius=28, thickness=2)
    draw_small_leaf_icon(frame, chart_x1 + 30, chart_y1 + 30, color=(85, 155, 70))
    draw_ui_text(frame, "Angle vs Time", chart_x1 + 54, chart_y1 + 40, scale=0.56, color=(30, 95, 45), thickness=2)

    if is_rom_movement:
        cv2.line(frame, (chart_x2 - 190, chart_y1 + 33), (chart_x2 - 166, chart_y1 + 33), (235, 120, 20), 3, cv2.LINE_AA)
        draw_ui_text_fit(frame, movement_label, chart_x2 - 158, chart_y1 + 40, 140, base_scale=0.36, min_scale=0.26, color=(45, 65, 75), thickness=1)
        draw_axis_line_chart(frame, samples, chart_x1 + 6, chart_y1 + 38, chart_x2 - 6, chart_y2 - 8, y_label="Angle (deg)")
    else:
        draw_axis_line_chart(
            frame,
            [],
            chart_x1 + 6,
            chart_y1 + 38,
            chart_x2 - 6,
            chart_y2 - 8,
            y_label="Angle (deg)",
            empty_title="Angle-time chart is not used for this movement",
            empty_subtitle="Use Accuracy, Error Details, and Progress Trend for calibration quality"
        )

    draw_donut_accuracy(frame, 800, 170, 1158, 325, summary, progress_selected_movement)
    draw_error_details_card(frame, 800, 340, 1158, 500, summary, progress_selected_movement)

    draw_progress_trend_compact(frame, progress_rows, progress_selected_movement, 115, 515, 965, 675, sample_rows)

    for button in buttons:
        if button["id"] in ["back_main", "movement_dropdown"] or button["id"].startswith("progress_"):
            continue
        disabled = button["id"] in ["session_prev", "session_next"] and len(sessions) <= 1
        draw_small_game_button(
            frame,
            button,
            hovered=(button["id"] == hover_button),
            selected=False,
            disabled=disabled,
            text_scale=0.43
        )

    draw_movement_dropdown_control(frame, buttons, hover_button)

    back_button = [button for button in buttons if button["id"] == "back_main"][0]
    draw_main_menu_button(frame, back_button, hovered=(back_button["id"] == hover_button))

    return frame

def progress_click_handler(clicked_button):
    global progress_selected_movement
    global progress_selected_session_index
    global progress_movement_dropdown_open
    global game_state

    if clicked_button is None:
        progress_movement_dropdown_open = False
        return

    if clicked_button == "back_main":
        progress_movement_dropdown_open = False
        game_state = "main_menu"
        return

    if clicked_button == "movement_dropdown":
        progress_movement_dropdown_open = not progress_movement_dropdown_open
        return

    rows = load_progress_rows_for_active_profile()
    sessions = build_analysis_sessions(rows)

    if clicked_button == "session_prev":
        progress_movement_dropdown_open = False
        if len(sessions) > 0:
            if progress_selected_session_index is None:
                progress_selected_session_index = len(sessions) - 1
            progress_selected_session_index = max(0, int(progress_selected_session_index) - 1)
        return

    if clicked_button == "session_next":
        progress_movement_dropdown_open = False
        if len(sessions) > 0:
            if progress_selected_session_index is None:
                progress_selected_session_index = len(sessions) - 1
            progress_selected_session_index = min(len(sessions) - 1, int(progress_selected_session_index) + 1)
        return

    if clicked_button.startswith("progress_"):
        selected = clicked_button.replace("progress_", "")
        if selected in ANALYSIS_ALL_MOVEMENT_KEYS:
            progress_selected_movement = selected
        progress_movement_dropdown_open = False
        return

    progress_movement_dropdown_open = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x
    global mouse_y
    global mouse_left_clicked

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_left_clicked = True


# -----------------------------
# Load assets
# -----------------------------
background = cv2.imread(BACKGROUND_PATH)

if background is None:
    print("Background image not found.")
    print("Make sure background_game.png is in the same folder.")
    exit()

background = cv2.resize(background, (WIDTH, HEIGHT))

main_menu_background_img = load_optional_menu_image(MAIN_MENU_BACKGROUND_PATH)
menu_panel_background_img = load_optional_menu_image(MENU_PANEL_BACKGROUND_PATH)
tutorial_background_img = load_optional_menu_image(TUTORIAL_BACKGROUND_PATH)
stage_select_background_img = load_first_optional_menu_image(BACKLEVEL_BACKGROUND_PATHS)

tree_stage1_img = load_tree_png_asset(TREE_STAGE1_PATH)
tree_stage2_img = load_tree_png_asset(TREE_STAGE2_PATH)
tree_stage3_img = load_tree_png_asset(TREE_STAGE3_PATH)

stage2_preview_img = load_optional_png_asset(STAGE2_PREVIEW_PATH)
stage3_preview_img = load_optional_png_asset(STAGE3_PREVIEW_PATH)
stage4_preview_img = load_optional_menu_image(STAGE4_PREVIEW_PATH)

stage2_background_img = load_optional_menu_image(STAGE2_BACKGROUND_PATH)
stage2_left_bush_stage1_img = load_optional_png_asset(STAGE2_LEFT_BUSH_STAGE1_PATH)
stage2_left_bush_stage2_img = load_optional_png_asset(STAGE2_LEFT_BUSH_STAGE2_PATH)
stage2_left_bush_stage3_img = load_optional_png_asset(STAGE2_LEFT_BUSH_STAGE3_PATH)
stage2_right_bush_stage1_img = load_optional_png_asset(STAGE2_RIGHT_BUSH_STAGE1_PATH)
stage2_right_bush_stage2_img = load_optional_png_asset(STAGE2_RIGHT_BUSH_STAGE2_PATH)
stage2_right_bush_stage3_img = load_optional_png_asset(STAGE2_RIGHT_BUSH_STAGE3_PATH)

stage3_background_img = load_optional_menu_image(STAGE3_BACKGROUND_PATH)
stage3_chrysanthemum_stage1_img = load_optional_png_asset(STAGE3_CHRYSANTHEMUM_STAGE1_PATH)
stage3_chrysanthemum_stage2_img = load_optional_png_asset(STAGE3_CHRYSANTHEMUM_STAGE2_PATH)
stage3_chrysanthemum_stage3_img = load_optional_png_asset(STAGE3_CHRYSANTHEMUM_STAGE3_PATH)
stage3_maple_stage1_img = load_optional_png_asset(STAGE3_MAPLE_STAGE1_PATH)
stage3_maple_stage2_img = load_optional_png_asset(STAGE3_MAPLE_STAGE2_PATH)
stage3_maple_stage3_img = load_optional_png_asset(STAGE3_MAPLE_STAGE3_PATH)
stage3_purple_bush_stage1_img = load_optional_png_asset(STAGE3_PURPLE_BUSH_STAGE1_PATH)
stage3_purple_bush_stage2_img = load_optional_png_asset(STAGE3_PURPLE_BUSH_STAGE2_PATH)
stage3_purple_bush_stage3_img = load_optional_png_asset(STAGE3_PURPLE_BUSH_STAGE3_PATH)

stage4_background_img = load_optional_menu_image(STAGE4_BACKGROUND_PATH)
stage4_winter_rose_stage1_img = load_optional_png_asset(STAGE4_WINTER_ROSE_STAGE1_PATH)
stage4_winter_rose_stage2_img = load_optional_png_asset(STAGE4_WINTER_ROSE_STAGE2_PATH)
stage4_winter_rose_stage3_img = load_optional_png_asset(STAGE4_WINTER_ROSE_STAGE3_PATH)
stage4_snowdrop_stage1_img = load_optional_png_asset(STAGE4_SNOWDROP_STAGE1_PATH)
stage4_snowdrop_stage2_img = load_optional_png_asset(STAGE4_SNOWDROP_STAGE2_PATH)
stage4_snowdrop_stage3_img = load_optional_png_asset(STAGE4_SNOWDROP_STAGE3_PATH)
stage4_poinsettia_stage1_img = load_optional_png_asset(STAGE4_POINSETTIA_STAGE1_PATH)
stage4_poinsettia_stage2_img = load_optional_png_asset(STAGE4_POINSETTIA_STAGE2_PATH)
stage4_poinsettia_stage3_img = load_optional_png_asset(STAGE4_POINSETTIA_STAGE3_PATH)
stage4_cyclamen_stage1_img = load_optional_png_asset(STAGE4_CYCLAMEN_STAGE1_PATH)
stage4_cyclamen_stage2_img = load_optional_png_asset(STAGE4_CYCLAMEN_STAGE2_PATH)
stage4_cyclamen_stage3_img = load_optional_png_asset(STAGE4_CYCLAMEN_STAGE3_PATH)

sun_frames = load_sun_frames()
cloud_frames = load_cloud_frames()

top_flower_stage1_img = load_png_asset(TOP_FLOWER_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
top_flower_stage2_img = load_png_asset(TOP_FLOWER_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
top_flower_stage3_img = load_png_asset(TOP_FLOWER_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

bottom_flower_stage1_img = load_png_asset(BOTTOM_FLOWER_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
bottom_flower_stage2_img = load_png_asset(BOTTOM_FLOWER_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
bottom_flower_stage3_img = load_png_asset(BOTTOM_FLOWER_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

right_orchid_stage1_img = load_png_asset(ORCHID_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
right_orchid_stage2_img = load_png_asset(ORCHID_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
right_orchid_stage3_img = load_png_asset(ORCHID_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

south_east_bluebloom_stage1_img = load_png_asset(BLUEBLOOM_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
south_east_bluebloom_stage2_img = load_png_asset(BLUEBLOOM_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
south_east_bluebloom_stage3_img = load_png_asset(BLUEBLOOM_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

left_tulip_stage1_img = load_png_asset(TULIP_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
left_tulip_stage2_img = load_png_asset(TULIP_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
left_tulip_stage3_img = load_png_asset(TULIP_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

south_west_peony_stage1_img = load_png_asset(PEONY_STAGE1_PATH, (FLOWER_SIZE, FLOWER_SIZE))
south_west_peony_stage2_img = load_png_asset(PEONY_STAGE2_PATH, (FLOWER_SIZE, FLOWER_SIZE))
south_west_peony_stage3_img = load_png_asset(PEONY_STAGE3_PATH, (FLOWER_SIZE, FLOWER_SIZE))

sun_animation = [
    (0, 0.70),
    (1, 0.22),
    (2, 0.22),
    (3, 0.22),
    (4, 0.90),
    (5, 0.22),
    (6, 0.22),
    (7, 0.22),
    (8, 0.90),
    (7, 0.22),
    (6, 0.22),
    (5, 0.22),
    (4, 0.90),
    (3, 0.22),
    (2, 0.22),
    (1, 0.22),
    (0, 0.90),
]

sun_animation_index = 0
sun_last_change_time = time.time()

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Cannot open camera.")
    exit()

cv2.namedWindow("Neck Rehab Game")
cv2.setMouseCallback("Neck Rehab Game", mouse_callback)


# -----------------------------
# Local data, active profile, progress session and music setup
# -----------------------------
ensure_data_dir()
current_profile = load_active_profile()
guest_stage_progress = get_default_stage_progress()
if current_profile is not None:
    current_profile["stage_progress"] = normalize_stage_progress(current_profile.get("stage_progress", {}))
    save_profile(current_profile)
profile_form_fields = default_profile_form_fields(current_profile)
profile_active_field_index = 0
profile_message = ""

MOVEMENT_DISPLAY_NAMES = {
    "flexion": "Flexion",
    "extension": "Extension",
    "left_bend": "Left Side Bend",
    "right_bend": "Right Side Bend",
    "chin_tuck": "Chin Tuck",
    "shoulder_lift": "Scapular Elevation",
    "palm_retraction": "Scapular Retraction",
}
progress_selected_movement = "flexion"
progress_selected_session_index = None  # None means latest saved session
progress_movement_dropdown_open = False

current_session = None
session_saved = False

settings_data = load_settings_data()
music_files = get_music_files()
music_available = False
music_volume = settings_data.get("volume", 0.45)
selected_music_index = settings_data.get("selected_music_index", None)
current_music_index = None
music_error_message = ""

if init_music_system() and len(music_files) > 0:
    start_random_music()


# -----------------------------
# Calibration variables
# -----------------------------
neutral_pitch = None
neutral_yaw = None
neutral_roll = None

# Neutral angle for left/right side bend
neutral_side_bend_angle = None

flexion_direction = None
flexion_threshold = None

extension_direction = None
extension_threshold = None

left_side_bend_direction = None
left_side_bend_threshold = None

right_side_bend_direction = None
right_side_bend_threshold = None

smoothed_pitch = None
smoothed_yaw = None
smoothed_roll = None

flexion_hold_start = None
extension_hold_start = None

left_side_bend_hold_start = None
right_side_bend_hold_start = None

# Chin Tuck variables
chin_neutral_features = None
chin_target_features = None
chin_neutral_pitch = None
chin_neutral_yaw = None
chin_neutral_eye_roll = None
chin_neutral_face_width = None

smoothed_chin_eye_roll = None

chin_feature_history = deque(maxlen=60)
chin_face_width_history = deque(maxlen=60)
chin_pitch_history = deque(maxlen=60)
chin_yaw_history = deque(maxlen=60)
chin_eye_roll_history = deque(maxlen=60)

chin_tuck_hold_start = None

# تایمرهای Chin Tuck مرحله گل
stage_chin_hold_start = None
rain_chin_hold_start = None

stage_chin_last_seen_time = None
rain_chin_last_seen_time = None

# بعد از تبدیل به ابر
rain_waiting_for_chin_release = False
cloud_activation_time = 0.0

# Scapular Elevation variables
shoulder_neutral_features = None
shoulder_target_features = None
shoulder_neutral_nose_y = None
shoulder_neutral_width = None
shoulder_neutral_angle = None

smoothed_shoulder_features = None
smoothed_shoulder_nose_y = None
smoothed_shoulder_width = None
smoothed_shoulder_angle = None

shoulder_feature_history = deque(maxlen=60)
shoulder_nose_y_history = deque(maxlen=60)
shoulder_width_history = deque(maxlen=60)
shoulder_angle_history = deque(maxlen=60)

shoulder_hold_start = None
shoulder_release_start_time = None
shoulder_toggle_waiting_release = False

# Scapular Retraction calibration variables
retraction_neutral_features = None
retraction_target_features = None
retraction_calibration_state = "capture_neutral"
retraction_neutral_start = None
retraction_target_start = None
retraction_release_start = None
retraction_test_hold_start = None
retraction_test_last_seen = None
retraction_calibration_message = "Show face, shoulders, and both palms. Keep palms outside shoulder width."
retraction_calibration_success = False

retraction_neutral_buffer = deque(maxlen=90)
retraction_target_buffer = deque(maxlen=90)
retraction_current_buffer = deque(maxlen=10)

locked_retraction_hold_start = None
locked_retraction_last_seen_time = None
locked_retraction_total_time = 0.0
locked_retraction_last_update_time = None

game_state = "main_menu"
game_finished = False
calibration_return_mode = "new_game"
pause_menu_enter_time = None
pause_return_state = "game"

print("Controls:")
print("SPACE = save neutral posture + Chin Tuck neutral + Scapular elevation neutral + Side Bend neutral")
print("F     = save forward flexion sample")
print("B     = save backward extension sample")
print("A     = save LEFT side bend sample")
print("D     = save RIGHT side bend sample")
print("T     = save Chin Tuck target")
print("U     = save Scapular Elevation target")
print("Auto  = Scapular Retraction calibration after U")
print("ENTER = start game after calibration")
print("R     = reset calibration")
print("Q     = quit")
print("")
print("Calibration order:")
print("SPACE -> F -> B -> A -> D -> Chin Tuck and T -> Scapular Elevation and U -> Auto Scapular Retraction -> ENTER")
while True:
    ret, cam_frame = cap.read()

    if not ret:
        break

    cam_frame = cv2.flip(cam_frame, 1)
    cam_h, cam_w, _ = cam_frame.shape

    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    pose_results = pose_detector.process(rgb)
    hand_results = hands_detector.process(rgb)

    current_pitch = None
    current_yaw = None
    current_roll = None
    current_eye_roll = None
    current_side_bend_angle = None

    current_chin_features = None
    current_chin_face_width = None

    current_shoulder_features = None
    current_shoulder_meta = None

    current_retraction_features = None
    retraction_face_center = None
    retraction_face_width = None
    retraction_left_hand = None
    retraction_right_hand = None
    retraction_shoulder_ref = None
    retraction_face_detected = False
    retraction_palms_detected = False
    retraction_shoulders_detected = False
    retraction_hands_outside_shoulders = False
    retraction_shoulder_gate_info = {
        "left_margin": -999.0,
        "right_margin": -999.0,
        "left_y_distance": 999.0,
        "right_y_distance": 999.0,
        "required_margin": RETRACTION_MIN_HAND_OUTSIDE_SHOULDER_MARGIN,
        "max_y_distance": RETRACTION_MAX_HAND_SHOULDER_Y_DISTANCE,
        "left_ok": False,
        "right_ok": False,
        "y_ok": False,
    }

    face_detected = False
    upper_body_detected = False

    # -----------------------------
    # FaceMesh processing
    # -----------------------------
    if results.multi_face_landmarks:
        face_detected = True
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        pose = get_head_pose(landmarks, cam_w, cam_h)

        if pose is not None:
            pitch, yaw, roll = pose

            smoothed_pitch = smooth_angle(pitch, smoothed_pitch, SMOOTHING)
            smoothed_yaw = smooth_angle(yaw, smoothed_yaw, SMOOTHING)
            smoothed_roll = smooth_angle(roll, smoothed_roll, SMOOTHING)

            current_pitch = smoothed_pitch
            current_yaw = smoothed_yaw
            current_roll = smoothed_roll

        raw_chin_features, raw_face_width = extract_chin_tuck_features(
            landmarks,
            cam_w,
            cam_h
        )

        # برای اینکه نویز لحظه‌ای باعث قطع و وصل شدن Chin Tuck نشود،
        # چند فریم آخر را میانگین می‌گیریم.
        chin_feature_history.append(raw_chin_features.copy())
        chin_face_width_history.append(raw_face_width)

        averaged_current_chin = average_recent_vectors(
            chin_feature_history,
            CHIN_CURRENT_AVERAGE_FRAMES
        )

        if averaged_current_chin is not None:
            current_chin_features = averaged_current_chin.copy()
        else:
            current_chin_features = raw_chin_features.copy()

        current_chin_face_width = raw_face_width

        raw_eye_roll = get_eye_roll(landmarks, cam_w, cam_h)

        smoothed_chin_eye_roll = smooth_angle(
            raw_eye_roll,
            smoothed_chin_eye_roll,
            CHIN_ANGLE_SMOOTHING
        )

        current_eye_roll = smoothed_chin_eye_roll
        current_side_bend_angle = smoothed_chin_eye_roll

        if current_pitch is not None:
            chin_pitch_history.append(current_pitch)

        if current_yaw is not None:
            chin_yaw_history.append(current_yaw)

        if current_eye_roll is not None:
            chin_eye_roll_history.append(current_eye_roll)

    # -----------------------------
    # Pose processing for Scapular Elevation
    # -----------------------------
    if pose_results.pose_landmarks:
        pose_landmarks = pose_results.pose_landmarks.landmark

        raw_shoulder_features, raw_shoulder_meta = extract_shoulder_lift_features(
            pose_landmarks,
            cam_w,
            cam_h
        )

        if raw_shoulder_features is not None:
            upper_body_detected = True

            if smoothed_shoulder_features is None:
                smoothed_shoulder_features = raw_shoulder_features.copy()
            else:
                smoothed_shoulder_features = (
                    SHOULDER_FEATURE_SMOOTHING * smoothed_shoulder_features +
                    (1 - SHOULDER_FEATURE_SMOOTHING) * raw_shoulder_features
                )

            current_shoulder_features = smoothed_shoulder_features.copy()

            smoothed_shoulder_nose_y = smooth_value(
                raw_shoulder_meta["nose_y"],
                smoothed_shoulder_nose_y,
                SHOULDER_FEATURE_SMOOTHING
            )

            smoothed_shoulder_width = smooth_value(
                raw_shoulder_meta["shoulder_width"],
                smoothed_shoulder_width,
                SHOULDER_FEATURE_SMOOTHING
            )

            smoothed_shoulder_angle = smooth_angle(
                raw_shoulder_meta["shoulder_angle"],
                smoothed_shoulder_angle,
                SHOULDER_ANGLE_SMOOTHING
            )

            current_shoulder_meta = {
                "nose_y": smoothed_shoulder_nose_y,
                "shoulder_width": smoothed_shoulder_width,
                "shoulder_angle": smoothed_shoulder_angle
            }

            shoulder_feature_history.append(current_shoulder_features.copy())
            shoulder_nose_y_history.append(current_shoulder_meta["nose_y"])
            shoulder_width_history.append(current_shoulder_meta["shoulder_width"])
            shoulder_angle_history.append(current_shoulder_meta["shoulder_angle"])


    # -----------------------------
    # Hands processing for Scapular Retraction
    # -----------------------------
    retraction_face_center, retraction_face_width = get_retraction_face_reference(
        results,
        cam_w,
        cam_h
    )

    retraction_left_hand, retraction_right_hand, retraction_hand_infos = get_two_retraction_screen_hands(
        hand_results,
        cam_w,
        cam_h
    )

    retraction_shoulder_ref = get_retraction_shoulder_reference(
        pose_results,
        cam_w,
        cam_h
    )

    retraction_face_detected = retraction_face_center is not None and retraction_face_width is not None
    retraction_palms_detected = retraction_left_hand is not None and retraction_right_hand is not None
    retraction_shoulders_detected = retraction_shoulder_ref is not None

    if retraction_palms_detected and retraction_face_detected:
        current_retraction_features = extract_retraction_features(
            retraction_left_hand,
            retraction_right_hand,
            retraction_face_center,
            retraction_face_width
        )

        if current_retraction_features is not None:
            retraction_current_buffer.append(current_retraction_features.copy())

        retraction_hands_outside_shoulders, retraction_shoulder_gate_info = retraction_hands_are_outside_shoulders(
            retraction_left_hand,
            retraction_right_hand,
            retraction_shoulder_ref,
            retraction_face_width
        )

    # -----------------------------
    # Character animation update
    # -----------------------------
    current_time = time.time()
    current_frame_id, current_duration = sun_animation[sun_animation_index]

    if current_time - sun_last_change_time >= current_duration:
        sun_animation_index = (sun_animation_index + 1) % len(sun_animation)
        sun_last_change_time = current_time
        current_frame_id, current_duration = sun_animation[sun_animation_index]

    current_sun = sun_frames[current_frame_id]
    current_cloud = cloud_frames[current_frame_id]

    update_music_playback()

    if game_state in ["tutorial", "stage2", "stage3", "stage4", "game"]:
        update_session_frame_metrics(
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle,
            current_chin_features,
            current_shoulder_features,
            current_retraction_features
        )

    # -----------------------------
    # Main menu screen
    # -----------------------------
    if game_state == "main_menu":
        frame = draw_main_menu_screen(current_sun, current_cloud)
        buttons = get_main_menu_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "start_game":
                start_new_game_from_main_menu()

            elif clicked_button == "create_profile":
                profile_form_fields = default_profile_form_fields(current_profile)
                profile_active_field_index = 0
                profile_message = ""
                game_state = "profile"

            elif clicked_button == "progress":
                game_state = "progress"

            elif clicked_button == "settings":
                game_state = "settings"

            elif clicked_button == "exit":
                quit_game = True

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Stage selection screen
    # -----------------------------
    elif game_state == "level_select":
        frame = draw_stage_select_screen()
        buttons = get_stage_select_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "back_main":
                stage_select_message = ""
                game_state = "main_menu"

            elif clicked_button is not None and clicked_button.startswith("stage_"):
                try:
                    chosen_stage = int(clicked_button.replace("stage_", ""))
                except Exception:
                    chosen_stage = None

                if chosen_stage is not None:
                    start_stage_calibration_from_select(chosen_stage)

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Profile screen
    # -----------------------------
    elif game_state == "profile":
        frame = draw_profile_screen()
        buttons = get_profile_buttons()

        if mouse_left_clicked:
            clicked_field = None
            x_left = 90
            x_right = 665
            y_start = 165
            field_w = 515
            field_h = 74
            gap = 18

            for i, field in enumerate(profile_form_fields):
                col_x = x_left if i < 5 else x_right
                row_i = i if i < 5 else i - 5
                y = y_start + row_i * (field_h + gap)
                if point_inside_rect(mouse_x, mouse_y, (col_x, y, col_x + field_w, y + field_h)):
                    clicked_field = i
                    break

            if clicked_field is not None:
                profile_active_field_index = clicked_field
            else:
                clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
                profile_click_handler(clicked_button)

            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Progress screen
    # -----------------------------
    elif game_state == "progress":
        frame = draw_progress_screen()
        buttons = get_progress_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            progress_click_handler(clicked_button)
            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Settings screen
    # -----------------------------
    elif game_state == "settings":
        frame = draw_settings_screen()
        buttons = get_settings_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            settings_click_handler(clicked_button)
            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Calibration screen
    # -----------------------------
    elif game_state == "calibration":
        # Calibration is now drawn on a full 1280x720 game-like layout:
        # section 1 = instruction panel, section 2 = saved steps, section 3 = webcam preview.
        calibration_camera = cam_frame.copy()

        if shoulder_target_features is not None and not retraction_calibration_success:
            update_auto_retraction_calibration(
                current_retraction_features,
                retraction_face_detected,
                retraction_palms_detected,
                retraction_shoulders_detected,
                retraction_hands_outside_shoulders,
                retraction_shoulder_gate_info
            )

            draw_retraction_calibration_guides(
                calibration_camera,
                hand_results,
                retraction_shoulder_ref,
                retraction_face_center,
                retraction_face_width,
                retraction_left_hand,
                retraction_right_hand,
                retraction_hands_outside_shoulders
            )

        frame = draw_calibration_screen(
            calibration_camera,
            face_detected,
            upper_body_detected,
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle
        )

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, get_calibration_buttons())
            mouse_left_clicked = False

            if clicked_button == "calibration_main_menu":
                finalize_session_save("calibration_main_menu")
                pause_menu_enter_time = None
                calibration_return_mode = "new_game"
                selected_recalibration_target = None
                game_state = "main_menu"

        cv2.imshow("Neck Rehab Game", frame)


    # -----------------------------
    # Easy tutorial stage
    # -----------------------------
    elif game_state == "tutorial":
        tutorial_status = process_tutorial_stage(
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle,
            current_chin_features
        )

        frame = draw_tutorial_screen(current_sun)

        if mouse_left_clicked:
            home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
            if home_hovered:
                enter_pause_menu("tutorial")
            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Easy tutorial win screen
    # -----------------------------
    elif game_state == "tutorial_win":
        frame = draw_tutorial_win_screen()
        buttons = get_tutorial_win_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "quit":
                quit_game = True

            elif clicked_button == "next_level":
                start_next_stage_from_completion()
                print("Next stage selected from Stage 1 complete screen.")

            elif clicked_button == "main_menu":
                finalize_session_save("tutorial_win_main_menu")
                pause_menu_enter_time = None
                calibration_return_mode = "new_game"
                selected_recalibration_target = None
                game_state = "main_menu"
                win_message = "Back to Main Menu selected."

        cv2.imshow("Neck Rehab Game", frame)


    # -----------------------------
    # Stage 2 - Summer Garden screen
    # -----------------------------
    elif game_state == "stage2":
        process_stage2_stage(
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle,
            current_chin_features,
            current_shoulder_features,
            current_shoulder_meta,
            current_retraction_features,
            retraction_face_detected,
            retraction_palms_detected,
            retraction_shoulders_detected,
            retraction_hands_outside_shoulders,
            retraction_shoulder_gate_info
        )

        frame = draw_stage2_screen(current_sun, current_cloud)

        if mouse_left_clicked:
            home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
            if home_hovered:
                enter_pause_menu("stage2")
            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)




    # -----------------------------
    # Stage 3 - Autumn Garden screen
    # -----------------------------
    elif game_state == "stage3":
        process_stage3_stage(
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle,
            current_chin_features,
            current_shoulder_features,
            current_shoulder_meta,
            current_retraction_features,
            retraction_face_detected,
            retraction_palms_detected,
            retraction_shoulders_detected,
            retraction_hands_outside_shoulders,
            retraction_shoulder_gate_info
        )
        frame = draw_stage3_screen(current_sun, current_cloud)
        if mouse_left_clicked:
            home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
            if home_hovered:
                enter_pause_menu("stage3")
            mouse_left_clicked = False
        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Stage 4 - Winter Garden screen
    # -----------------------------
    elif game_state == "stage4":
        process_stage4_stage(
            current_pitch,
            current_yaw,
            current_roll,
            current_side_bend_angle,
            current_chin_features,
            current_shoulder_features,
            current_shoulder_meta,
            current_retraction_features,
            retraction_face_detected,
            retraction_palms_detected,
            retraction_shoulders_detected,
            retraction_hands_outside_shoulders,
            retraction_shoulder_gate_info
        )
        frame = draw_stage4_screen(current_sun, current_cloud)
        if mouse_left_clicked:
            home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
            if home_hovered:
                enter_pause_menu("stage4")
            mouse_left_clicked = False
        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Main game screen
    # -----------------------------
    elif game_state == "game":
        frame = background.copy()

        top_flower_stage, top_flower_animating, top_flower_progress = update_flower_growth(
            top_flower_stage,
            top_flower_animating,
            top_flower_start_time
        )

        bottom_flower_stage, bottom_flower_animating, bottom_flower_progress = update_flower_growth(
            bottom_flower_stage,
            bottom_flower_animating,
            bottom_flower_start_time
        )

        right_orchid_stage, right_orchid_animating, right_orchid_progress = update_flower_growth(
            right_orchid_stage,
            right_orchid_animating,
            right_orchid_start_time
        )

        south_east_bluebloom_stage, south_east_bluebloom_animating, south_east_bluebloom_progress = update_flower_growth(
            south_east_bluebloom_stage,
            south_east_bluebloom_animating,
            south_east_bluebloom_start_time
        )

        left_tulip_stage, left_tulip_animating, left_tulip_progress = update_flower_growth(
            left_tulip_stage,
            left_tulip_animating,
            left_tulip_start_time
        )

        south_west_peony_stage, south_west_peony_animating, south_west_peony_progress = update_flower_growth(
            south_west_peony_stage,
            south_west_peony_animating,
            south_west_peony_start_time
        )

        top_flower_img = get_flower_asset(
            top_flower_stage,
            top_flower_stage1_img,
            top_flower_stage2_img,
            top_flower_stage3_img
        )

        bottom_flower_img = get_flower_asset(
            bottom_flower_stage,
            bottom_flower_stage1_img,
            bottom_flower_stage2_img,
            bottom_flower_stage3_img
        )

        right_orchid_img = get_flower_asset(
            right_orchid_stage,
            right_orchid_stage1_img,
            right_orchid_stage2_img,
            right_orchid_stage3_img
        )

        south_east_bluebloom_img = get_flower_asset(
            south_east_bluebloom_stage,
            south_east_bluebloom_stage1_img,
            south_east_bluebloom_stage2_img,
            south_east_bluebloom_stage3_img
        )

        left_tulip_img = get_flower_asset(
            left_tulip_stage,
            left_tulip_stage1_img,
            left_tulip_stage2_img,
            left_tulip_stage3_img
        )

        south_west_peony_img = get_flower_asset(
            south_west_peony_stage,
            south_west_peony_stage1_img,
            south_west_peony_stage2_img,
            south_west_peony_stage3_img
        )

        if top_flower_img is not None:
            frame = draw_flower_on_pot(
                frame,
                top_flower_img,
                TOP_POT_CENTER_X,
                TOP_POT_SOIL_Y,
                top_flower_progress
            )

        if bottom_flower_img is not None:
            frame = draw_flower_on_pot(
                frame,
                bottom_flower_img,
                BOTTOM_POT_CENTER_X,
                BOTTOM_POT_SOIL_Y,
                bottom_flower_progress
            )

        if right_orchid_img is not None:
            frame = draw_flower_on_pot(
                frame,
                right_orchid_img,
                RIGHT_ORCHID_POT_CENTER_X,
                RIGHT_ORCHID_POT_SOIL_Y,
                right_orchid_progress
            )

        if south_east_bluebloom_img is not None:
            frame = draw_flower_on_pot(
                frame,
                south_east_bluebloom_img,
                SOUTH_EAST_BLUEBLOOM_POT_CENTER_X,
                SOUTH_EAST_BLUEBLOOM_POT_SOIL_Y,
                south_east_bluebloom_progress
            )

        if left_tulip_img is not None:
            frame = draw_flower_on_pot(
                frame,
                left_tulip_img,
                LEFT_TULIP_POT_CENTER_X,
                LEFT_TULIP_POT_SOIL_Y,
                left_tulip_progress
            )

        if south_west_peony_img is not None:
            frame = draw_flower_on_pot(
                frame,
                south_west_peony_img,
                SOUTH_WEST_PEONY_POT_CENTER_X,
                SOUTH_WEST_PEONY_POT_SOIL_Y,
                south_west_peony_progress
            )

        movement_status = "Move gently forward or backward"
        shoulder_status = ""
        rain_status = ""

        # فعلاً باران با Chin Tuck ابر غیرفعال است.
        # در مرحله بعدی، این متغیر با حرکت عقب بردن شانه/کتف فعال می‌شود.
        rain_visual_active = False

        # -------------------------------------------------
        # State separation
        # -------------------------------------------------
        # 1) اگر گل Stage 1 باشد، فقط Chin Tuck برای Stage 2 کار می‌کند.
        # 2) در حالت ابر، Chin Tuck کاملاً غیرفعال است و هیچ اتفاقی ایجاد نمی‌کند.
        # 3) در غیر این حالت، Flexion/Extension و Shoulder Toggle کار می‌کنند.
        waiting_for_stage_chin = (
            (active_flower == "top" and top_flower_stage == 1) or
            (active_flower == "bottom" and bottom_flower_stage == 1)
        )

        # -----------------------------
        # Step 3 lock mode: Stage 1 Chin Tuck -> Stage 2 Scapular Elevation -> Cloud
        # -----------------------------
        # وقتی خورشید بالای گل قفل است، حرکت‌های جابه‌جایی غیرفعال می‌مانند.
        # Stage 1: فقط Chin Tuck تجمعی ۱۰ ثانیه
        # Stage 2: فقط Scapular Elevation دو ثانیه، سپس تبدیل به ابر در همان نقطه
        if character_locked_to_flower:
            clear_all_movement_holds()

            locked_name = get_flower_name(locked_flower_key)
            locked_stage = get_flower_stage_value(locked_flower_key)

            rain_sequence_status = update_locked_rain_sequence()

            if rain_sequence_status is not None:
                reset_locked_chin_tuck_progress()
                reset_locked_shoulder_lift_progress()
                reset_locked_retraction_progress()
                movement_status = rain_sequence_status

            elif locked_stage == 1:
                reset_locked_shoulder_lift_progress()

                movement_status = (
                    f"Only Chin Tuck: {locked_chin_tuck_total_time:.1f}s / "
                    f"{LOCKED_CHIN_REQUIRED_TOTAL_TIME:.1f}s"
                )

                if (
                    current_chin_features is not None and
                    chin_neutral_features is not None and
                    chin_target_features is not None
                ):
                    is_chin_tuck, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                        current_chin_features,
                        chin_neutral_features,
                        chin_target_features,
                        current_pitch,
                        neutral_pitch,
                        current_yaw,
                        neutral_yaw,
                        current_roll,
                        neutral_roll
                    )

                    total_chin_time = update_locked_chin_tuck_progress(is_chin_tuck)

                    if is_chin_tuck:
                        movement_status = (
                            f"Chin Tuck total: {total_chin_time:.1f}s / "
                            f"{LOCKED_CHIN_REQUIRED_TOTAL_TIME:.1f}s | "
                            f"score: {chin_score:.2f} | progress: {chin_progress:.2f}"
                        )
                    else:
                        movement_status = (
                            f"Only Chin Tuck allowed | total: {total_chin_time:.1f}s / "
                            f"{LOCKED_CHIN_REQUIRED_TOTAL_TIME:.1f}s | "
                            f"not chin tuck | score: {chin_score:.2f} | progress: {chin_progress:.2f}"
                        )

                    if total_chin_time >= LOCKED_CHIN_REQUIRED_TOTAL_TIME:
                        set_flower_stage2(locked_flower_key)
                        reset_locked_chin_tuck_progress()
                        reset_locked_shoulder_lift_progress()

                        movement_status = (
                            f"Chin Tuck complete! {locked_name} is now Stage 2. "
                            f"Now only Scapular Elevation for 5s is allowed."
                        )

                else:
                    locked_chin_tuck_last_update_time = None
                    movement_status = (
                        f"Only Chin Tuck allowed | total: {locked_chin_tuck_total_time:.1f}s / "
                        f"{LOCKED_CHIN_REQUIRED_TOTAL_TIME:.1f}s | keep face visible"
                    )

            elif locked_stage == 2:
                reset_locked_chin_tuck_progress()

                if active_character == "sun":
                    movement_status = process_locked_shoulder_to_cloud(
                        current_pitch,
                        current_yaw,
                        current_roll,
                        current_shoulder_features,
                        current_shoulder_meta
                    )
                else:
                    reset_locked_shoulder_lift_progress()
                    movement_status = process_locked_cloud_retraction_rain(
                        current_retraction_features,
                        retraction_face_detected,
                        retraction_palms_detected,
                        retraction_shoulders_detected,
                        retraction_hands_outside_shoulders,
                        retraction_shoulder_gate_info
                    )

            else:
                reset_locked_chin_tuck_progress()
                reset_locked_shoulder_lift_progress()

                movement_status = (
                    f"Locked above {locked_name} | Stage {locked_stage}. "
                    f"Next step will be added later."
                )

        # -----------------------------
        # 1) Chin Tuck: Stage 1 -> Stage 2
        # -----------------------------
        elif waiting_for_stage_chin:
            flexion_hold_start = None
            extension_hold_start = None
            shoulder_hold_start = None
            shoulder_release_start_time = None
            shoulder_toggle_waiting_release = False

            rain_chin_hold_start = None
            rain_chin_last_seen_time = None

            movement_status = "Hold Chin Tuck for 3 seconds"

            if (
                current_chin_features is not None and
                chin_neutral_features is not None and
                chin_target_features is not None
            ):
                is_chin_tuck, chin_score, target_strength, current_strength, chin_progress, chin_side_error = is_simple_chin_tuck(
                    current_chin_features,
                    chin_neutral_features,
                    chin_target_features,
                    current_pitch,
                    neutral_pitch,
                    current_yaw,
                    neutral_yaw,
                    current_roll,
                    neutral_roll
                )

                stage_chin_hold_start, stage_chin_last_seen_time, chin_hold_time, chin_hold_active, chin_noise_ignored = update_chin_hold_with_tolerance(
                    is_chin_tuck,
                    stage_chin_hold_start,
                    stage_chin_last_seen_time
                )

                if chin_hold_active:
                    if chin_noise_ignored:
                        movement_status = (
                            f"Chin Tuck: {chin_hold_time:.1f}s / 3.0s "
                            f"| noise ignored | score: {chin_score:.2f} | progress: {chin_progress:.2f}"
                        )
                    else:
                        movement_status = (
                            f"Chin Tuck: {chin_hold_time:.1f}s / 3.0s "
                            f"| score: {chin_score:.2f} | progress: {chin_progress:.2f}"
                        )

                    if chin_hold_time >= CHIN_REQUIRED_HOLD_TIME:
                        if active_flower == "top" and top_flower_stage == 1:
                            top_flower_stage = 2
                            score += 1
                            movement_status = "Chin Tuck success! Top flower Stage 2."

                        elif active_flower == "bottom" and bottom_flower_stage == 1:
                            bottom_flower_stage = 2
                            score += 1
                            movement_status = "Chin Tuck success! Red rose Stage 2."

                        sun_shining_start_time = time.time()

                        stage_chin_hold_start = None
                        stage_chin_last_seen_time = None
                        active_flower = None

                else:
                    stage_chin_hold_start = None
                    stage_chin_last_seen_time = None

                    movement_status = (
                        f"Not Chin Tuck | score: {chin_score:.2f} "
                        f"| progress: {chin_progress:.2f} | side: {chin_side_error:.2f} "
                        f"| target: {target_strength:.3f} | current: {current_strength:.3f}"
                    )

        # -----------------------------
        # 2) Cloud mode without Chin Tuck
        # -----------------------------
        # در حالت ابر، Chin Tuck هیچ کاری انجام نمی‌دهد.
        # باران بعداً با حرکت جداگانه عقب بردن شانه/کتف اضافه می‌شود.

        # -----------------------------
        # 3) Free movement mode:
        # Flexion / Extension + Side Bend + Shoulder Toggle
        # -----------------------------
        else:
            stage_chin_hold_start = None
            stage_chin_last_seen_time = None

            rain_chin_hold_start = None
            rain_chin_last_seen_time = None
            rain_waiting_for_chin_release = False

            # First check side-bend movement.
            # If side bend is active, do not process flexion/extension in the same moment.
            side_bend_status = process_side_bend_movement(
                current_side_bend_angle,
                current_pitch,
                current_yaw
            )

            if side_bend_status != "":
                movement_status = side_bend_status
                flexion_hold_start = None
                extension_hold_start = None

            else:
                # Flexion moves forward/down; Extension moves backward/up.
                if current_pitch is not None and neutral_pitch is not None:
                    pitch_delta = angle_diff(current_pitch, neutral_pitch)
                    yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))

                    if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
                        movement_status = "Wrong movement: do not turn left/right."
                        flexion_hold_start = None
                        extension_hold_start = None

                    else:
                        flexion_amount = flexion_direction * pitch_delta
                        extension_amount = extension_direction * pitch_delta

                        flexion_detected = False
                        extension_detected = False

                        if flexion_amount >= flexion_threshold:
                            if flexion_hold_start is None:
                                flexion_hold_start = time.time()

                            if time.time() - flexion_hold_start >= FLEXION_REQUIRED_HOLD_TIME:
                                flexion_detected = True
                        else:
                            flexion_hold_start = None

                        if extension_amount >= extension_threshold:
                            if extension_hold_start is None:
                                extension_hold_start = time.time()

                            if time.time() - extension_hold_start >= EXTENSION_REQUIRED_HOLD_TIME:
                                extension_detected = True
                        else:
                            extension_hold_start = None

                        if flexion_detected:
                            movement_status = "FLEXION DETECTED - Character moves down"

                            if time.time() - last_sun_move_time >= SUN_MOVE_COOLDOWN:
                                candidate_y = min(
                                    SUN_MAX_Y,
                                    sun_target_y + SUN_MOVE_DISTANCE
                                )

                                if not can_move_character_to(sun_target_x, candidate_y):
                                    movement_status = "Cannot move farther down."
                                    flexion_hold_start = None

                                else:
                                    sun_target_y = candidate_y

                                    last_sun_move_time = time.time()
                                    flexion_hold_start = None

                                    side_pot_message = check_side_pot_reached()
                                    if side_pot_message != "":
                                        movement_status = side_pot_message

                                    # All six pots, including the center-bottom pot,
                                    # are handled by check_side_pot_reached().

                        elif extension_detected:
                            movement_status = "EXTENSION DETECTED - Character moves up"

                            if time.time() - last_sun_move_time >= SUN_MOVE_COOLDOWN:
                                candidate_y = max(
                                    SUN_MIN_Y,
                                    sun_target_y - SUN_MOVE_DISTANCE
                                )

                                if not can_move_character_to(sun_target_x, candidate_y):
                                    movement_status = "Cannot move farther up."
                                    extension_hold_start = None

                                else:
                                    sun_target_y = candidate_y

                                    last_sun_move_time = time.time()
                                    extension_hold_start = None

                                    side_pot_message = check_side_pot_reached()
                                    if side_pot_message != "":
                                        movement_status = side_pot_message

                                    # All six pots, including the center-top pot,
                                    # are handled by check_side_pot_reached().

            # Scapular Elevation toggle
            shoulder_status = process_shoulder_toggle(
                current_pitch,
                current_yaw,
                current_roll,
                current_shoulder_features,
                current_shoulder_meta
            )

        # -----------------------------
        # Smooth character movement
        # -----------------------------
        sun_current_x += (sun_target_x - sun_current_x) * 0.22
        sun_current_y += (sun_target_y - sun_current_y) * 0.22

        # خورشید فقط وقتی می‌درخشد که در حالت خورشید باشیم
        # و Chin Tuck مرحله گل فعال باشد یا تازه موفق شده باشد.
        sun_is_shining = (
            active_character == "sun" and
            (
                stage_chin_hold_start is not None or
                locked_chin_tuck_last_update_time is not None or
                (time.time() - sun_shining_start_time) <= SUN_SHINING_DURATION
            )
        )

        # باران با حرکت عقب بردن شانه/کتف فعال می‌شود و برای چند ثانیه روی همان گل نمایش داده می‌شود.
        rain_is_active = (
            (time.time() - rain_effect_start_time) <= RAIN_EFFECT_DURATION
        )

        if rain_is_active:
            frame = draw_rain(
                frame,
                int(rain_effect_x),
                int(rain_effect_y),
                SUN_SIZE
            )

        if active_character == "sun":
            if sun_is_shining:
                frame = draw_sun_glow(
                    frame,
                    int(sun_current_x),
                    int(sun_current_y),
                    SUN_SIZE
                )

            current_character_frame = current_sun

        else:
            current_character_frame = current_cloud

        frame = overlay_transparent(
            frame,
            current_character_frame,
            int(sun_current_x),
            int(sun_current_y)
        )

        draw_text(frame, f"Score: {score}", 40, 60, 0.9)
        draw_text(frame, movement_status, 40, 110, 0.62)

        text_y = 150

        if shoulder_status != "":
            draw_text(frame, shoulder_status, 40, text_y, 0.55, (255, 255, 255))
            text_y += 40

        if rain_status != "":
            draw_text(frame, rain_status, 40, text_y, 0.55, (255, 255, 255))
            text_y += 40

        draw_text(frame, f"Character: {active_character}", 40, text_y, 0.55, (255, 255, 0))
        draw_text(frame, "Q: quit | R: recalibrate", 40, text_y + 40, 0.58)

        home_hovered = point_inside_rect(mouse_x, mouse_y, get_home_button_rect())
        frame = draw_home_icon_button(frame, hovered=home_hovered)

        if mouse_left_clicked:
            if home_hovered:
                enter_pause_menu()
            mouse_left_clicked = False

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Home / Pause menu
    # -----------------------------
    elif game_state == "pause_menu":
        frame = draw_pause_menu_screen()
        buttons = get_pause_menu_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "continue":
                resume_game_from_pause()

            elif clicked_button == "recalibrate":
                start_recalibration_from_pause_menu()

            elif clicked_button == "main_menu":
                finalize_session_save("returned_to_main_menu")
                pause_menu_enter_time = None
                calibration_return_mode = "new_game"
                game_state = "main_menu"
                win_message = "Main Menu selected from Home Menu."

            elif clicked_button == "quit":
                finalize_session_save("quit")
                quit_game = True

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Recalibration movement selection
    # -----------------------------
    elif game_state == "recalibrate_select":
        frame = draw_recalibrate_selection_screen()
        buttons = get_recalibrate_selection_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "back_pause":
                game_state = "pause_menu"

            elif clicked_button is not None and clicked_button.startswith("recalibrate_"):
                selected_target = clicked_button.replace("recalibrate_", "")
                start_selected_recalibration_from_menu(selected_target)

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Win screen
    # -----------------------------
    elif game_state == "win":
        frame = background.copy()
        frame = draw_win_screen(frame)

        buttons = get_win_buttons()

        if mouse_left_clicked:
            clicked_button = get_button_at_position(mouse_x, mouse_y, buttons)
            mouse_left_clicked = False

            if clicked_button == "quit":
                quit_game = True

            elif clicked_button == "main_menu":
                finalize_session_save("win_main_menu")
                game_state = "main_menu"
                win_message = "Back to Main Menu selected."

            elif clicked_button == "next_level":
                start_next_stage_from_completion()
                win_message = "Next Stage selected."

        cv2.imshow("Neck Rehab Game", frame)

    # -----------------------------
    # Future Next Level placeholder
    # -----------------------------
    elif game_state == "next_level":
        frame = draw_placeholder_screen(
            "Next Level",
            "Next Level was clicked."
        )
        cv2.imshow("Neck Rehab Game", frame)

            # -----------------------------
    # Key controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    if game_state == "profile":
        handle_profile_keyboard(key)
        if quit_game:
            break
        continue

    if game_state in ["main_menu", "level_select", "progress", "settings", "recalibrate_select", "tutorial_win"]:
        if key in [ord("q"), ord("Q")]:
            finalize_session_save("quit")
            break
        if key == 27:
            if game_state == "recalibrate_select":
                game_state = "pause_menu"
            else:
                game_state = "main_menu"
        if quit_game:
            finalize_session_save("quit")
            break
        continue

    if key in [ord("q"), ord("Q")]:
        finalize_session_save("quit")
        break

    elif key in [ord("r"), ord("R")]:
        finalize_session_save("reset")
        neutral_pitch = None
        neutral_yaw = None
        neutral_roll = None
        neutral_side_bend_angle = None

        flexion_direction = None
        flexion_threshold = None

        extension_direction = None
        extension_threshold = None

        left_side_bend_direction = None
        left_side_bend_threshold = None

        right_side_bend_direction = None
        right_side_bend_threshold = None

        smoothed_pitch = None
        smoothed_yaw = None
        smoothed_roll = None

        flexion_hold_start = None
        extension_hold_start = None

        left_side_bend_hold_start = None
        right_side_bend_hold_start = None

        sun_current_x = float(sun_x)
        sun_current_y = float(sun_y)

        sun_target_x = float(sun_x)
        sun_target_y = float(sun_y)

        last_sun_move_time = 0
        sun_shining_start_time = 0
        rain_effect_start_time = 0
        rain_effect_x = float(sun_x)
        rain_effect_y = float(sun_y)
        cloud_activation_time = 0.0

        active_character = "sun"

        score = 0

        top_flower_stage = 0
        bottom_flower_stage = 0

        right_orchid_stage = 0
        south_east_bluebloom_stage = 0
        left_tulip_stage = 0
        south_west_peony_stage = 0

        top_flower_animating = False
        bottom_flower_animating = False

        right_orchid_animating = False
        south_east_bluebloom_animating = False
        left_tulip_animating = False
        south_west_peony_animating = False

        top_flower_start_time = 0
        bottom_flower_start_time = 0

        right_orchid_start_time = 0
        south_east_bluebloom_start_time = 0
        left_tulip_start_time = 0
        south_west_peony_start_time = 0

        active_flower = None
        character_locked_to_flower = False
        locked_flower_key = None
        reset_locked_chin_tuck_progress()
        reset_locked_shoulder_lift_progress()
        reached_side_pots = set()

        chin_neutral_features = None
        chin_target_features = None
        chin_neutral_pitch = None
        chin_neutral_yaw = None
        chin_neutral_eye_roll = None
        chin_neutral_face_width = None

        smoothed_chin_eye_roll = None

        clear_chin_histories()

        chin_tuck_hold_start = None

        stage_chin_hold_start = None
        rain_chin_hold_start = None

        stage_chin_last_seen_time = None
        rain_chin_last_seen_time = None

        rain_waiting_for_chin_release = False

        shoulder_neutral_features = None
        shoulder_target_features = None
        shoulder_neutral_nose_y = None
        shoulder_neutral_width = None
        shoulder_neutral_angle = None

        smoothed_shoulder_features = None
        smoothed_shoulder_nose_y = None
        smoothed_shoulder_width = None
        smoothed_shoulder_angle = None

        clear_shoulder_histories()

        shoulder_hold_start = None
        shoulder_release_start_time = None
        shoulder_toggle_waiting_release = False

        reset_retraction_calibration_state(clear_saved=True)
        reset_locked_retraction_progress()
        reset_locked_rain_sequence()
        if game_state == "stage2":
            reset_stage2_state_keep_calibration()
            selected_stage_number = 2
            calibration_return_mode = "stage_start"
        elif game_state == "stage3":
            reset_stage3_state_keep_calibration()
            selected_stage_number = 3
            calibration_return_mode = "stage_start"
        elif game_state == "stage4":
            reset_stage4_state_keep_calibration()
            selected_stage_number = 4
            calibration_return_mode = "stage_start"

        game_state = "calibration"
        game_finished = False
        win_message = ""
        mouse_left_clicked = False
        calibration_return_mode = "new_game"
        if current_stage_number in [1, 2, 3, 4, 5]:
            selected_stage_number = current_stage_number
            calibration_return_mode = "stage_start"
        pause_menu_enter_time = None

        start_new_session_metrics("reset_calibration")
        print("Calibration reset.")

    elif key == ord(" "):
        if (
            current_pitch is not None and
            current_side_bend_angle is not None and
            current_chin_features is not None and
            current_shoulder_features is not None and
            current_shoulder_meta is not None
        ):
            neutral_pitch = current_pitch
            neutral_yaw = current_yaw
            neutral_roll = current_roll
            neutral_side_bend_angle = current_side_bend_angle

            smoothed_pitch = current_pitch
            smoothed_yaw = current_yaw
            smoothed_roll = current_roll

            flexion_direction = None
            flexion_threshold = None

            extension_direction = None
            extension_threshold = None

            left_side_bend_direction = None
            left_side_bend_threshold = None

            right_side_bend_direction = None
            right_side_bend_threshold = None

            flexion_hold_start = None
            extension_hold_start = None

            left_side_bend_hold_start = None
            right_side_bend_hold_start = None

            averaged_chin_features = average_recent_vectors(
                chin_feature_history,
                CHIN_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_chin_face_width = average_recent_values(
                chin_face_width_history,
                CHIN_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_chin_pitch = average_recent_values(
                chin_pitch_history,
                CHIN_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_chin_yaw = average_recent_values(
                chin_yaw_history,
                CHIN_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_chin_eye_roll = average_recent_values(
                chin_eye_roll_history,
                CHIN_NEUTRAL_AVERAGE_FRAMES
            )

            if averaged_chin_features is not None:
                chin_neutral_features = averaged_chin_features.copy()
            else:
                chin_neutral_features = current_chin_features.copy()

            if averaged_chin_face_width is not None:
                chin_neutral_face_width = averaged_chin_face_width
            else:
                chin_neutral_face_width = current_chin_face_width

            if averaged_chin_pitch is not None:
                chin_neutral_pitch = averaged_chin_pitch
            else:
                chin_neutral_pitch = current_pitch

            if averaged_chin_yaw is not None:
                chin_neutral_yaw = averaged_chin_yaw
            else:
                chin_neutral_yaw = current_yaw

            if averaged_chin_eye_roll is not None:
                chin_neutral_eye_roll = averaged_chin_eye_roll
            else:
                chin_neutral_eye_roll = current_eye_roll

            chin_target_features = None

            averaged_shoulder_features = average_recent_vectors(
                shoulder_feature_history,
                SHOULDER_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_shoulder_nose_y = average_recent_values(
                shoulder_nose_y_history,
                SHOULDER_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_shoulder_width = average_recent_values(
                shoulder_width_history,
                SHOULDER_NEUTRAL_AVERAGE_FRAMES
            )

            averaged_shoulder_angle = average_recent_values(
                shoulder_angle_history,
                SHOULDER_NEUTRAL_AVERAGE_FRAMES
            )

            if averaged_shoulder_features is not None:
                shoulder_neutral_features = averaged_shoulder_features.copy()
            else:
                shoulder_neutral_features = current_shoulder_features.copy()

            if averaged_shoulder_nose_y is not None:
                shoulder_neutral_nose_y = averaged_shoulder_nose_y
            else:
                shoulder_neutral_nose_y = current_shoulder_meta["nose_y"]

            if averaged_shoulder_width is not None:
                shoulder_neutral_width = averaged_shoulder_width
            else:
                shoulder_neutral_width = current_shoulder_meta["shoulder_width"]

            if averaged_shoulder_angle is not None:
                shoulder_neutral_angle = averaged_shoulder_angle
            else:
                shoulder_neutral_angle = current_shoulder_meta["shoulder_angle"]

            shoulder_target_features = None

            clear_chin_histories()
            clear_shoulder_histories()

            chin_tuck_hold_start = None

            stage_chin_hold_start = None
            rain_chin_hold_start = None

            stage_chin_last_seen_time = None
            rain_chin_last_seen_time = None

            shoulder_hold_start = None
            shoulder_release_start_time = None
            shoulder_toggle_waiting_release = False

            reset_retraction_calibration_state(clear_saved=True)
            reset_locked_retraction_progress()
            reset_locked_rain_sequence()

            print("Neutral saved.")
            print(f"Pitch = {neutral_pitch:.2f}")
            print(f"Yaw   = {neutral_yaw:.2f}")
            print(f"Roll  = {neutral_roll:.2f}")
            print("Chin Tuck neutral saved.")
            print("Scapular elevation neutral saved.")
            print("Side Bend neutral saved.")
            print(f"Side Bend angle = {neutral_side_bend_angle:.2f}")

        else:
            print("Face/shoulders are not ready. Make sure head and both shoulders are visible.")

    elif key in [ord("f"), ord("F")]:
        if current_pitch is not None and neutral_pitch is not None:
            pitch_delta = angle_diff(current_pitch, neutral_pitch)
            yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))

            if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
                print("Too much left/right turn. Keep face forward and bend down only.")

            elif abs(pitch_delta) < MIN_FLEXION_SAMPLE_DELTA:
                print("Movement is too small. Bend forward/down a little more, then press F.")

            else:
                flexion_direction = 1 if pitch_delta > 0 else -1

                sample_amount = abs(pitch_delta)

                flexion_threshold = max(
                    MIN_FLEXION_THRESHOLD,
                    sample_amount * FLEXION_THRESHOLD_RATIO
                )

                print("Flexion sample saved.")
                print(f"Sample pitch delta = {pitch_delta:.2f}")
                print(f"Flexion direction = {flexion_direction}")
                print(f"Flexion threshold = {flexion_threshold:.2f}")

    elif key in [ord("b"), ord("B")]:
        if current_pitch is not None and neutral_pitch is not None:
            pitch_delta = angle_diff(current_pitch, neutral_pitch)
            yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))

            if yaw_delta > MAX_ALLOWED_YAW_CHANGE:
                print("Too much left/right turn. Keep face forward and move backward only.")

            elif abs(pitch_delta) < MIN_EXTENSION_SAMPLE_DELTA:
                print("Movement is too small. Move backward a little more, then press B.")

            else:
                new_extension_direction = 1 if pitch_delta > 0 else -1

                if flexion_direction is not None and new_extension_direction == flexion_direction:
                    print("Extension sample looks like the same direction as Flexion.")
                    print("Move your head backward more clearly, then press B again.")

                else:
                    extension_direction = new_extension_direction

                    sample_amount = abs(pitch_delta)

                    extension_threshold = max(
                        MIN_EXTENSION_THRESHOLD,
                        sample_amount * EXTENSION_THRESHOLD_RATIO
                    )

                    print("Extension sample saved.")
                    print(f"Sample pitch delta = {pitch_delta:.2f}")
                    print(f"Extension direction = {extension_direction}")
                    print(f"Extension threshold = {extension_threshold:.2f}")

    elif key in [ord("a"), ord("A")]:
        if (
            current_side_bend_angle is not None and
            neutral_side_bend_angle is not None and
            current_pitch is not None and
            neutral_pitch is not None and
            current_yaw is not None and
            neutral_yaw is not None
        ):
            side_bend_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
            pitch_delta = abs(angle_diff(current_pitch, neutral_pitch))
            yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))

            if pitch_delta > MAX_ALLOWED_PITCH_CHANGE_FOR_SIDE_BEND:
                print("Too much forward/backward movement. Bend only toward LEFT shoulder.")

            elif yaw_delta > MAX_ALLOWED_YAW_CHANGE_FOR_SIDE_BEND:
                print("Too much face rotation. Keep face forward and bend toward LEFT shoulder.")

            elif abs(side_bend_delta) < MIN_SIDE_BEND_SAMPLE_DELTA:
                print("Left side bend movement is too small.")
                print("Bend your head more clearly toward LEFT shoulder, then press A again.")

            else:
                left_side_bend_direction = 1 if side_bend_delta > 0 else -1

                sample_amount = abs(side_bend_delta)

                left_side_bend_threshold = max(
                    MIN_SIDE_BEND_THRESHOLD,
                    sample_amount * SIDE_BEND_THRESHOLD_RATIO
                )

                left_side_bend_hold_start = None
                right_side_bend_hold_start = None

                print("Left Side Bend sample saved.")
                print(f"Sample side bend delta = {side_bend_delta:.2f}")
                print(f"Left side bend direction = {left_side_bend_direction}")
                print(f"Left side bend threshold = {left_side_bend_threshold:.2f}")

        else:
            print("First press SPACE for neutral calibration.")

    elif key in [ord("d"), ord("D")]:
        if (
            current_side_bend_angle is not None and
            neutral_side_bend_angle is not None and
            current_pitch is not None and
            neutral_pitch is not None and
            current_yaw is not None and
            neutral_yaw is not None
        ):
            side_bend_delta = angle_diff(current_side_bend_angle, neutral_side_bend_angle)
            pitch_delta = abs(angle_diff(current_pitch, neutral_pitch))
            yaw_delta = abs(angle_diff(current_yaw, neutral_yaw))

            if pitch_delta > MAX_ALLOWED_PITCH_CHANGE_FOR_SIDE_BEND:
                print("Too much forward/backward movement. Bend only toward RIGHT shoulder.")

            elif yaw_delta > MAX_ALLOWED_YAW_CHANGE_FOR_SIDE_BEND:
                print("Too much face rotation. Keep face forward and bend toward RIGHT shoulder.")

            elif abs(side_bend_delta) < MIN_SIDE_BEND_SAMPLE_DELTA:
                print("Right side bend movement is too small.")
                print("Bend your head more clearly toward RIGHT shoulder, then press D again.")

            else:
                new_right_direction = 1 if side_bend_delta > 0 else -1

                if left_side_bend_direction is not None and new_right_direction == left_side_bend_direction:
                    print("Right Side Bend looks like the same direction as Left Side Bend.")
                    print("Bend your head to the opposite side more clearly, then press D again.")

                else:
                    right_side_bend_direction = new_right_direction

                    sample_amount = abs(side_bend_delta)

                    right_side_bend_threshold = max(
                        MIN_SIDE_BEND_THRESHOLD,
                        sample_amount * SIDE_BEND_THRESHOLD_RATIO
                    )

                    left_side_bend_hold_start = None
                    right_side_bend_hold_start = None

                    print("Right Side Bend sample saved.")
                    print(f"Sample side bend delta = {side_bend_delta:.2f}")
                    print(f"Right side bend direction = {right_side_bend_direction}")
                    print(f"Right side bend threshold = {right_side_bend_threshold:.2f}")

        else:
            print("First press SPACE for neutral calibration.")

    elif key in [ord("t"), ord("T")]:
        if current_chin_features is not None and chin_neutral_features is not None:
            averaged_target = average_recent_vectors(
                chin_feature_history,
                CHIN_TARGET_AVERAGE_FRAMES
            )

            if averaged_target is not None:
                target_candidate = averaged_target.copy()
            else:
                target_candidate = current_chin_features.copy()

            test_strength = float(
                np.linalg.norm(target_candidate - chin_neutral_features)
            )

            if test_strength < CHIN_MIN_TARGET_STRENGTH:
                print("Chin Tuck target movement is too small.")
                print("Pull chin backward more clearly, hold it for 1 second, then press T again.")
                print(f"Target strength = {test_strength:.4f}")
                print(f"Minimum required = {CHIN_MIN_TARGET_STRENGTH:.4f}")

            else:
                chin_target_features = target_candidate.copy()

                chin_tuck_hold_start = None

                stage_chin_hold_start = None
                rain_chin_hold_start = None

                stage_chin_last_seen_time = None
                rain_chin_last_seen_time = None

                clear_chin_histories()

                print("Chin Tuck target saved.")
                print(f"Target strength = {test_strength:.4f}")
                print("Target change:", chin_target_features - chin_neutral_features)

        else:
            print("First press SPACE for neutral calibration.")

    elif key in [ord("u"), ord("U")]:
        if current_shoulder_features is not None and shoulder_neutral_features is not None:
            averaged_shoulder_target = average_recent_vectors(
                shoulder_feature_history,
                SHOULDER_TARGET_AVERAGE_FRAMES
            )

            if averaged_shoulder_target is not None:
                target_candidate = averaged_shoulder_target.copy()
            else:
                target_candidate = current_shoulder_features.copy()

            test_strength = float(
                np.linalg.norm(
                    (target_candidate - shoulder_neutral_features) *
                    SHOULDER_FEATURE_WEIGHTS
                )
            )

            left_target_lift = target_candidate[0] - shoulder_neutral_features[0]
            right_target_lift = target_candidate[1] - shoulder_neutral_features[1]

            if test_strength < SHOULDER_MIN_TARGET_STRENGTH:
                print("Scapular Elevation target movement is too small.")
                print("Lift both shoulders more clearly, hold 1 second, then press U again.")
                print(f"Target strength = {test_strength:.4f}")

            elif left_target_lift < SHOULDER_MIN_SINGLE_LIFT and right_target_lift < SHOULDER_MIN_SINGLE_LIFT:
                print("Shoulders did not lift enough.")
                print("Lift shoulders more clearly, hold 1 second, then press U again.")
                print(f"Left target lift = {left_target_lift:.4f}")
                print(f"Right target lift = {right_target_lift:.4f}")

            else:
                shoulder_target_features = target_candidate.copy()

                shoulder_hold_start = None
                shoulder_release_start_time = None
                shoulder_toggle_waiting_release = False

                clear_shoulder_histories()
                reset_retraction_calibration_state(clear_saved=True)

                print("Scapular Elevation target saved.")
                print(f"Target strength = {test_strength:.4f}")
                print(f"Left target lift = {left_target_lift:.4f}")
                print(f"Right target lift = {right_target_lift:.4f}")

        else:
            print("First press SPACE for neutral calibration.")

    elif key == 13:
        if (
            neutral_pitch is not None
            and neutral_side_bend_angle is not None
            and flexion_threshold is not None
            and extension_threshold is not None
            and left_side_bend_threshold is not None
            and right_side_bend_threshold is not None
            and chin_neutral_features is not None
            and chin_target_features is not None
            and shoulder_neutral_features is not None
            and shoulder_target_features is not None
            and retraction_neutral_features is not None
            and retraction_target_features is not None
            and retraction_calibration_success
        ):
            if calibration_return_mode == "resume_game":
                if current_session is None:
                    start_new_session_metrics("resume_after_recalibration", stage_number=current_stage_number)
                resume_game_from_pause()
                game_finished = False
                win_message = ""

                stage_chin_hold_start = None
                rain_chin_hold_start = None

                stage_chin_last_seen_time = None
                rain_chin_last_seen_time = None

                shoulder_hold_start = None
                shoulder_release_start_time = None
                shoulder_toggle_waiting_release = False
                pause_locked_shoulder_lift_progress()
                pause_locked_retraction_progress()
                pause_stage2_shoulder_progress()
                pause_stage2_retraction_progress()
                pause_stage3_shoulder_progress()
                pause_stage3_retraction_progress()
                pause_stage4_shoulder_progress()
                pause_stage4_retraction_progress()
                selected_recalibration_target = None

                print("Recalibration complete. Game continued from the same progress.")

            else:
                if current_session is None:
                    reason = f"stage_{selected_stage_number}_calibration_complete" if selected_stage_number is not None else "calibration_complete"
                    start_new_session_metrics(reason)

                if calibration_return_mode == "stage_start":
                    start_selected_stage_after_calibration()
                else:
                    start_tutorial_stage_after_calibration()

                game_finished = False
                win_message = ""

                stage_chin_hold_start = None
                rain_chin_hold_start = None

                stage_chin_last_seen_time = None
                rain_chin_last_seen_time = None

                shoulder_hold_start = None
                shoulder_release_start_time = None
                shoulder_toggle_waiting_release = False
                reset_locked_retraction_progress()
                calibration_return_mode = "new_game"
                pause_menu_enter_time = None
                selected_recalibration_target = None

                print("Selected stage started after calibration.")

        else:
            print("Calibration is not complete yet.")
            print("Need: SPACE neutral, F flexion, B extension, A left bend, D right bend, T chin tuck target, U scapular elevation target, Auto Scapular Retraction calibration.")

    if quit_game:
        break


finalize_session_save("exit")

if music_available and pygame is not None:
    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception:
        pass

cap.release()
face_mesh.close()
pose_detector.close()
hands_detector.close()
cv2.destroyAllWindows()
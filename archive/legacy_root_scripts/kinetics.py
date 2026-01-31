import csv
import os

import datasets


_DESCRIPTION = "Kinetics dataset"
_HOMEPAGE = "https://www.deepmind.com/open-source/kinetics"
_CITATION = """
@misc{https://doi.org/10.48550/arxiv.1705.06950,
  doi = {10.48550/ARXIV.1705.06950},
  url = {https://arxiv.org/abs/1705.06950},
  author = {Kay, Will and Carreira, Joao and Simonyan, Karen and Zhang, Brian and Hillier, Chloe and Vijayanarasimhan, Sudheendra and Viola, Fabio and Green, Tim and Back, Trevor and Natsev, Paul and Suleyman, Mustafa and Zisserman, Andrew},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {The Kinetics Human Action Video Dataset},
  publisher = {arXiv},
  year = {2017},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
""".strip()

_LICENSE = "cc-by-4.0"

_NAMES = ['abseiling', 'air_drumming', 'answering_questions', 'applauding', 'applying_cream', 'archery', 'arm_wrestling', 'arranging_flowers', 'assembling_computer', 'auctioning', 'baby_waking_up', 'baking_cookies', 'balloon_blowing', 'bandaging', 'barbequing', 'bartending', 'beatboxing', 'bee_keeping', 'belly_dancing', 'bench_pressing', 'bending_back', 'bending_metal', 'biking_through_snow', 'blasting_sand', 'blowing_glass', 'blowing_leaves', 'blowing_nose', 'blowing_out_candles', 'bobsledding', 'bookbinding', 'bouncing_on_trampoline', 'bowling', 'braiding_hair', 'breading_or_breadcrumbing', 'breakdancing', 'brush_painting', 'brushing_hair', 'brushing_teeth', 'building_cabinet', 'building_shed', 'bungee_jumping', 'busking', 'canoeing_or_kayaking', 'capoeira', 'carrying_baby', 'cartwheeling', 'carving_pumpkin', 'catching_fish', 'catching_or_throwing_baseball', 'catching_or_throwing_frisbee', 'catching_or_throwing_softball', 'celebrating', 'changing_oil', 'changing_wheel', 'checking_tires', 'cheerleading', 'chopping_wood', 'clapping', 'clay_pottery_making', 'clean_and_jerk', 'cleaning_floor', 'cleaning_gutters', 'cleaning_pool', 'cleaning_shoes', 'cleaning_toilet', 'cleaning_windows', 'climbing_a_rope', 'climbing_ladder', 'climbing_tree', 'contact_juggling', 'cooking_chicken', 'cooking_egg', 'cooking_on_campfire', 'cooking_sausages', 'counting_money', 'country_line_dancing', 'cracking_neck', 'crawling_baby', 'crossing_river', 'crying', 'curling_hair', 'cutting_nails', 'cutting_pineapple', 'cutting_watermelon', 'dancing_ballet', 'dancing_charleston', 'dancing_gangnam_style', 'dancing_macarena', 'deadlifting', 'decorating_the_christmas_tree', 'digging', 'dining', 'disc_golfing', 'diving_cliff', 'dodgeball', 'doing_aerobics', 'doing_laundry', 'doing_nails', 'drawing', 'dribbling_basketball', 'drinking', 'drinking_beer', 'drinking_shots', 'driving_car', 'driving_tractor', 'drop_kicking', 'drumming_fingers', 'dunking_basketball', 'dying_hair', 'eating_burger', 'eating_cake', 'eating_carrots', 'eating_chips', 'eating_doughnuts', 'eating_hotdog', 'eating_ice_cream', 'eating_spaghetti', 'eating_watermelon', 'egg_hunting', 'exercising_arm', 'exercising_with_an_exercise_ball', 'extinguishing_fire', 'faceplanting', 'feeding_birds', 'feeding_fish', 'feeding_goats', 'filling_eyebrows', 'finger_snapping', 'fixing_hair', 'flipping_pancake', 'flying_kite', 'folding_clothes', 'folding_napkins', 'folding_paper', 'front_raises', 'frying_vegetables', 'garbage_collecting', 'gargling', 'getting_a_haircut', 'getting_a_tattoo', 'giving_or_receiving_award', 'golf_chipping', 'golf_driving', 'golf_putting', 'grinding_meat', 'grooming_dog', 'grooming_horse', 'gymnastics_tumbling', 'hammer_throw', 'headbanging', 'headbutting', 'high_jump', 'high_kick', 'hitting_baseball', 'hockey_stop', 'holding_snake', 'hopscotch', 'hoverboarding', 'hugging', 'hula_hooping', 'hurdling', 'hurling_sport', 'ice_climbing', 'ice_fishing', 'ice_skating', 'ironing', 'javelin_throw', 'jetskiing', 'jogging', 'juggling_balls', 'juggling_fire', 'juggling_soccer_ball', 'jumping_into_pool', 'jumpstyle_dancing', 'kicking_field_goal', 'kicking_soccer_ball', 'kissing', 'kitesurfing', 'knitting', 'krumping', 'laughing', 'laying_bricks', 'long_jump', 'lunge', 'making_a_cake', 'making_a_sandwich', 'making_bed', 'making_jewelry', 'making_pizza', 'making_snowman', 'making_sushi', 'making_tea', 'marching', 'massaging_back', 'massaging_feet', 'massaging_legs', 'massaging_persons_head', 'milking_cow', 'mopping_floor', 'motorcycling', 'moving_furniture', 'mowing_lawn', 'news_anchoring', 'opening_bottle', 'opening_present', 'paragliding', 'parasailing', 'parkour', 'passing_American_football_in_game', 'passing_American_football_not_in_game', 'peeling_apples', 'peeling_potatoes', 'petting_animal_not_cat', 'petting_cat', 'picking_fruit', 'planting_trees', 'plastering', 'playing_accordion', 'playing_badminton', 'playing_bagpipes', 'playing_basketball', 'playing_bass_guitar', 'playing_cards', 'playing_cello', 'playing_chess', 'playing_clarinet', 'playing_controller', 'playing_cricket', 'playing_cymbals', 'playing_didgeridoo', 'playing_drums', 'playing_flute', 'playing_guitar', 'playing_harmonica', 'playing_harp', 'playing_ice_hockey', 'playing_keyboard', 'playing_kickball', 'playing_monopoly', 'playing_organ', 'playing_paintball', 'playing_piano', 'playing_poker', 'playing_recorder', 'playing_saxophone', 'playing_squash_or_racquetball', 'playing_tennis', 'playing_trombone', 'playing_trumpet', 'playing_ukulele', 'playing_violin', 'playing_volleyball', 'playing_xylophone', 'pole_vault', 'presenting_weather_forecast', 'pull_ups', 'pumping_fist', 'pumping_gas', 'punching_bag', 'punching_person_boxing', 'push_up', 'pushing_car', 'pushing_cart', 'pushing_wheelchair', 'reading_book', 'reading_newspaper', 'recording_music', 'riding_a_bike', 'riding_camel', 'riding_elephant', 'riding_mechanical_bull', 'riding_mountain_bike', 'riding_mule', 'riding_or_walking_with_horse', 'riding_scooter', 'riding_unicycle', 'ripping_paper', 'robot_dancing', 'rock_climbing', 'rock_scissors_paper', 'roller_skating', 'running_on_treadmill', 'sailing', 'salsa_dancing', 'sanding_floor', 'scrambling_eggs', 'scuba_diving', 'setting_table', 'shaking_hands', 'shaking_head', 'sharpening_knives', 'sharpening_pencil', 'shaving_head', 'shaving_legs', 'shearing_sheep', 'shining_shoes', 'shooting_basketball', 'shooting_goal_soccer', 'shot_put', 'shoveling_snow', 'shredding_paper', 'shuffling_cards', 'side_kick', 'sign_language_interpreting', 'singing', 'situp', 'skateboarding', 'ski_jumping', 'skiing_crosscountry', 'skiing_not_slalom_or_crosscountry', 'skiing_slalom', 'skipping_rope', 'skydiving', 'slacklining', 'slapping', 'sled_dog_racing', 'smoking', 'smoking_hookah', 'snatch_weight_lifting', 'sneezing', 'sniffing', 'snorkeling', 'snowboarding', 'snowkiting', 'snowmobiling', 'somersaulting', 'spinning_poi', 'spray_painting', 'spraying', 'springboard_diving', 'squat', 'sticking_tongue_out', 'stomping_grapes', 'stretching_arm', 'stretching_leg', 'strumming_guitar', 'surfing_crowd', 'surfing_water', 'sweeping_floor', 'swimming_backstroke', 'swimming_breast_stroke', 'swimming_butterfly_stroke', 'swing_dancing', 'swinging_legs', 'swinging_on_something', 'sword_fighting', 'tai_chi', 'taking_a_shower', 'tango_dancing', 'tap_dancing', 'tapping_guitar', 'tapping_pen', 'tasting_beer', 'tasting_food', 'testifying', 'texting', 'throwing_axe', 'throwing_ball', 'throwing_discus', 'tickling', 'tobogganing', 'tossing_coin', 'tossing_salad', 'training_dog', 'trapezing', 'trimming_or_shaving_beard', 'trimming_trees', 'triple_jump', 'tying_bow_tie', 'tying_knot_not_on_a_tie', 'tying_tie', 'unboxing', 'unloading_truck', 'using_computer', 'using_remote_controller_not_gaming', 'using_segway', 'vault', 'waiting_in_line', 'walking_the_dog', 'washing_dishes', 'washing_feet', 'washing_hair', 'washing_hands', 'water_skiing', 'water_sliding', 'watering_plants', 'waxing_back', 'waxing_chest', 'waxing_eyebrows', 'waxing_legs', 'weaving_basket', 'welding', 'whistling', 'windsurfing', 'wrapping_present', 'wrestling', 'writing', 'yawning', 'yoga', 'zumba']


_TAR_URLS = {
    'train': [f'https://s3.amazonaws.com/kinetics/400/train/part_{i}.tar.gz' for i in range(242)], # 242
    'val': [f'https://s3.amazonaws.com/kinetics/400/val/part_{i}.tar.gz' for i in range(20)] # 20
}

_ANNOTATION_URLS = {
    'train': 'https://s3.amazonaws.com/kinetics/400/annotations/train.csv',
    'val': 'https://s3.amazonaws.com/kinetics/400/annotations/val.csv'
}


class Kinetics(datasets.GeneratorBasedBuilder):
    """Kinetics 400 Video Action Recognition dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "video": datasets.Value('binary'),
                    "label": datasets.ClassLabel(names=_NAMES),
                    "time_start": datasets.Value('int32'),
                    "time_end": datasets.Value('int32'),
                    "is_cc": datasets.Value('bool')
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        archive_paths = dl_manager.download(_TAR_URLS)
        annotation_files = dl_manager.download_and_extract(_ANNOTATION_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": [dl_manager.iter_archive(x) for x in archive_paths['train']],
                    "annotation_file": annotation_files["train"],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": [dl_manager.iter_archive(x) for x in archive_paths['val']],
                    "annotation_file": annotation_files["val"],
                },
            ),
        ]

    def _generate_examples(self, files, annotation_file):
        """Generate videos and labels for splits."""

        annotations = {}
        file_fmtstr = "{ytid}_{start:06}_{end:06}.mp4"
        with open(annotation_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                f = file_fmtstr.format(
                    ytid=row["youtube_id"],
                    start=int(row["time_start"]),
                    end=int(row["time_end"]),
                )
                label = row["label"].replace(" ", "_").replace("'", "").replace("(", "").replace(")", "")
                annotations[f] = {'label': label, 'time_start': row['time_start'], 'time_end': row['time_end'], 'is_cc': row['is_cc']}

        for archive in files:
            for file_path, file_obj in archive:
                filename = os.path.basename(file_path)
                if filename in annotations:
                    data = annotations[filename]
                    data['video'] = file_obj.read()
                    yield filename[:-4], data

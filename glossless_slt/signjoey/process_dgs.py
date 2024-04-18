import tensorflow_datasets as tfds
import re
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
import os

def load_dgs():
  os.environ['TFDS_DATA_DIR'] = 'D:\\tensorflow_datasets'
  config = DgsCorpusConfig(name="sentences-openpose", include_video=False, include_pose="openpose", data_type="sentence")
  dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config))

  return (dgs_corpus)

GLOSSES_TO_IGNORE = ["$GEST-OFF", "$$EXTRA-LING-MAN"]


def collapse_gloss(gloss: str) -> str:
    """
    Collapse phonological variations of the same type, and
    - for number signs remove handshape variants
    - keep numerals ($NUM), list glosses ($LIST) and finger alphabet ($ALPHA)
    :param gloss:
    :return:
    """
    try:
        collapsed_gloss_groups = re.search(r"([$A-Z-ÖÄÜ]+[0-9]*)[A-Z]*(:?[0-9A-ZÖÄÜ]*o?f?[0-9]*)", gloss).groups()
        collapsed_gloss = "".join([g for g in collapsed_gloss_groups if g is not None])
    except AttributeError:
        print("Gloss could not be generalized: '%s'", gloss)
        collapsed_gloss = gloss

    return collapsed_gloss


def generalize_dgs_glosses(glosses) -> str:
    """
    This code is taken from:
    https://github.com/bricksdont/easier-gloss-translation/blob/gloss_preprocessing_2/scripts/preprocessing/preprocess_glosses.py

    Removes certain kinds of variation in order to bolster generalization.
    Example:
    ICH1 ETWAS-PLANEN-UND-UMSETZEN1 SELBST1A* KLAPPT1* $GEST-OFF^ BIS-JETZT1 GEWOHNHEIT1* $GEST-OFF^*
    becomes:
    ICH1 ETWAS-PLANEN-UND-UMSETZEN1 SELBST1 KLAPPT1 BIS-JETZT1 GEWOHNHEIT1
    :param line:
    :return:
    """
  
    collapsed_glosses = []

    for gloss in glosses:
        
        gloss = gloss.strip()
        
        # remove ad-hoc deviations from citation forms
        gloss = gloss.replace("*", "")

        # remove distinction between type glosses and subtype glosses
        gloss = gloss.replace("^", "")
        collapsed_gloss = collapse_gloss(gloss)

        # remove special glosses that cannot possibly help translation
        if collapsed_gloss in GLOSSES_TO_IGNORE:
            continue

        collapsed_glosses.append(collapsed_gloss)

    return collapsed_glosses
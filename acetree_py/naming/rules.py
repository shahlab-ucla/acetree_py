"""Division rules for the Sulston naming system.

Each Rule describes how a parent cell divides: which axis the division
occurs along, and what the two daughter cells should be named.

Rules come from two sources:
  1. new_rules.tsv (NewRules.txt) — pre-computed rules with empirical axis vectors
     for ~620 known cell divisions.
  2. names_hash.csv (namesHash.txt) — a lookup of parent → sulston_letter for
     ~60 cells that need non-default naming.

For unknown parents not in either table, a default rule is generated based
on the last character of the parent name.

Ported from: DivisionCaller.Rule, DivisionCaller.readNewRules(),
             DivisionCaller.readSulstonRules()
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass

import numpy as np

from .sulston_names import LETTER_TO_AXIS, complement

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A division rule describing how a parent cell divides.

    Attributes:
        parent: Parent cell name (e.g. "AB").
        sulston_letter: The Sulston letter for daughter1 (e.g. "a").
        daughter1: First daughter name (e.g. "ABa") — the "positive dot product" side.
        daughter2: Second daughter name (e.g. "ABp").
        axis_vector: 3D unit vector for the expected division axis.
    """
    parent: str
    sulston_letter: str
    daughter1: str
    daughter2: str
    axis_vector: np.ndarray


def _load_new_rules() -> dict[str, Rule]:
    """Load pre-computed rules from new_rules.tsv (NewRules.txt).

    Format: Parent<TAB>Rule<TAB>D1<TAB>D2<TAB>X<TAB>Y<TAB>Z
    The 'Rule' column is always 0 in the current data (unused).
    """
    rules: dict[str, Rule] = {}
    try:
        ref = importlib.resources.files("acetree_py.resources").joinpath("new_rules.tsv")
        text = ref.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Could not load new_rules.tsv; no pre-computed rules available")
        return rules

    for line_num, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("Parent"):
            continue  # skip header or empty

        parts = line.split("\t")
        if len(parts) < 7:
            logger.debug("Skipping malformed line %d in new_rules.tsv: %r", line_num, line)
            continue

        parent = parts[0].strip()
        # parts[1] is the rule code (always 0, unused)
        dau1 = parts[2].strip()
        dau2 = parts[3].strip()
        try:
            x = float(parts[4])
            y = float(parts[5])
            z = float(parts[6])
        except ValueError:
            logger.debug("Skipping line %d with bad floats in new_rules.tsv", line_num)
            continue

        # Determine sulston letter from daughter1 name
        sulston_letter = dau1[-1] if dau1 else "a"

        rules[parent] = Rule(
            parent=parent,
            sulston_letter=sulston_letter,
            daughter1=dau1,
            daughter2=dau2,
            axis_vector=np.array([x, y, z], dtype=np.float64),
        )

    logger.info("Loaded %d pre-computed division rules", len(rules))
    return rules


def _load_names_hash() -> dict[str, str]:
    """Load the Sulston name hash from names_hash.csv.

    Format: parent_name,encoded_value
    The encoded value contains the sulston letter and axis info.

    Returns:
        Dict mapping parent_name -> sulston_letter.
    """
    lookup: dict[str, str] = {}
    try:
        ref = importlib.resources.files("acetree_py.resources").joinpath("names_hash.csv")
        text = ref.read_text(encoding="utf-8")
    except Exception:
        logger.warning("Could not load names_hash.csv; no sulston name hash available")
        return lookup

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) < 2:
            continue

        parent = parts[0].strip()
        encoded = parts[1].strip()

        # Decode the encoded value to extract the sulston letter.
        # Format examples: "i0ya" -> letter is encoded in the string
        # The Java code parses this as:
        #   first char: 'a'=anterior, 'i'=inferior (default 'a')
        #   then '0'
        #   then axis: 'x'=AP, 'y'=LR, 'z'=DV
        #   then qualifier: 'a'=anterior side, 'p'=posterior side, 'r'=reverse
        # But the actual sulston letter derivation is:
        #   For the Java namesHash, the key insight is:
        #   - The encoded string tells us which letter to use for the FIRST daughter
        #   - "a0zp" -> z-axis, p-side -> "a" (anterior for z) -> sulston_letter = "a"
        #   - "i0ya" -> y-axis, a-side -> "l" (left for y) -> sulston_letter = "l"
        #   - "l0"   -> direct letter "l"
        #   - "d0"   -> direct letter "d"
        #   - "a0xa" -> x-axis, a-side -> sulston_letter = "a"
        # Simplified: if encoded is just 2 chars like "l0" or "d0", the letter is encoded[0]
        # Otherwise we need to decode the axis mapping.
        if len(encoded) == 2 and encoded[1] == "0":
            # Direct letter: "l0" -> "l", "d0" -> "d"
            letter = encoded[0]
        elif len(encoded) >= 4:
            # Encoded format: [orientation][0][axis][qualifier]
            axis_char = encoded[2] if len(encoded) > 2 else "x"
            qualifier = encoded[3] if len(encoded) > 3 else "a"

            # Map axis + qualifier to sulston letter
            if axis_char == "x":
                # x-axis = AP: a-side -> "a", p-side -> "p"
                letter = "a" if qualifier in ("a", "r") else "p"
            elif axis_char == "y":
                # y-axis = LR: a-side -> "l", p-side -> "r", r-reverse
                if qualifier == "a":
                    letter = "l"
                elif qualifier == "r":
                    letter = "r"
                else:
                    letter = "l"
            elif axis_char == "z":
                # z-axis = DV: a-side -> "d" (in Java, z maps to "a" for AP-like)
                # Actually in Java DivisionCaller, z-axis means the vector is [0,0,1]
                # and the default letter for unknown z-axis is "a"
                letter = "a" if qualifier in ("a", "r") else "p"
            else:
                letter = "a"
        else:
            letter = "a"

        lookup[parent] = letter

    logger.info("Loaded %d Sulston name hash entries", len(lookup))
    return lookup


class RuleManager:
    """Manages division rules, providing lookups and generating defaults.

    This replaces the DivisionCaller's rule management in Java.
    Rules are looked up in priority order:
      1. Pre-computed rules from NewRules.txt (empirical axis vectors)
      2. Generated rules using namesHash.txt lookup + default axis vectors
      3. Auto-generated rules based on the parent's last naming letter
    """

    def __init__(self) -> None:
        self._new_rules = _load_new_rules()
        self._names_hash = _load_names_hash()
        self._generated_rules: dict[str, Rule] = {}

    def get_rule(self, parent_name: str) -> Rule:
        """Get or create a division rule for the given parent.

        Args:
            parent_name: The name of the dividing cell.

        Returns:
            The Rule describing how this cell should divide.
        """
        # 1. Check pre-computed rules
        if parent_name in self._new_rules:
            return self._new_rules[parent_name]

        # 2. Check cached generated rules
        if parent_name in self._generated_rules:
            return self._generated_rules[parent_name]

        # 3. Generate a new rule
        rule = self._generate_rule(parent_name)
        self._generated_rules[parent_name] = rule
        return rule

    def _generate_rule(self, parent_name: str) -> Rule:
        """Generate a rule for an unknown parent.

        Uses namesHash lookup if available, otherwise defaults based on
        the last character of the parent name.
        """
        # Look up sulston letter from names hash
        if parent_name in self._names_hash:
            letter = self._names_hash[parent_name]
        else:
            # Default: "a" (anterior/posterior division)
            letter = "a"

        # Build daughter names
        comp = complement(letter)
        dau1 = parent_name + letter
        dau2 = parent_name + comp

        # Build axis vector from letter
        axis = LETTER_TO_AXIS.get(letter, (1.0, 0.0, 0.0))
        axis_vec = np.array(axis, dtype=np.float64)

        return Rule(
            parent=parent_name,
            sulston_letter=letter,
            daughter1=dau1,
            daughter2=dau2,
            axis_vector=axis_vec,
        )

    @property
    def num_precomputed(self) -> int:
        """Number of pre-computed rules loaded from NewRules.txt."""
        return len(self._new_rules)

    @property
    def num_hash_entries(self) -> int:
        """Number of entries loaded from namesHash.txt."""
        return len(self._names_hash)

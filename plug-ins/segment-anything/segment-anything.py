#!/usr/bin/env python3
# Segment Anything for GIMP
# Copyright (C) 2026  SAM-GIMP-Plugin contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GIMP 3 plug-in entry point.

Registers two menu items:

  Select > Segment Anything...          the segmentation window
  Filters > Segment Anything > Setup... install or change the model

The heavy lifting happens in the ``samgimp`` package next to this file and
in a helper process that owns its own Python environment.
"""

import os
import sys

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
from gi.repository import Gimp, GimpUi, GLib  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from samgimp import env, log  # noqa: E402

PROC_SEGMENT = "python-fu-segment-anything"
PROC_SETUP = "python-fu-segment-anything-setup"

AUTHOR = "SAM-GIMP-Plugin contributors"
COPYRIGHT_YEAR = "2026"


def _fail(procedure, message):
    error = GLib.Error.new_literal(Gimp.PlugIn.error_quark(), message, 0)
    return procedure.new_return_values(Gimp.PDBStatusType.EXECUTION_ERROR, error)


def _cancel(procedure):
    return procedure.new_return_values(Gimp.PDBStatusType.CANCEL, GLib.Error())


def _success(procedure):
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


def run_segment(procedure, run_mode, image, drawables, config, run_data):
    if run_mode != Gimp.RunMode.INTERACTIVE:
        return _fail(procedure,
                     "Segment Anything needs its window; run it interactively.")
    try:
        GimpUi.init(PROC_SEGMENT)
        from samgimp import dialog_main, gimpio

        gimpio.cleanup_temp()
        settings = env.load_settings()
        dialog_main.run(image, list(drawables or []), settings)
        return _success(procedure)
    except Exception as error:  # noqa: BLE001 - report instead of crashing GIMP
        log.exception("segment procedure failed")
        return _fail(procedure, "Segment Anything failed: %s" % error)


def run_setup(procedure, run_mode, image, drawables, config, run_data):
    if run_mode != Gimp.RunMode.INTERACTIVE:
        return _fail(procedure, "Setup needs its window; run it interactively.")
    try:
        GimpUi.init(PROC_SETUP)
        from samgimp import dialog_setup

        dialog_setup.run_setup(parent=None, settings=env.load_settings())
        return _success(procedure)
    except Exception as error:  # noqa: BLE001
        log.exception("setup procedure failed")
        return _fail(procedure, "Segment Anything setup failed: %s" % error)


class SegmentAnything(Gimp.PlugIn):
    """Plug-in registration."""

    def do_set_i18n(self, procname):
        # The plug-in ships no translation catalogue; saying so keeps GIMP
        # from logging a warning for every procedure at start-up.
        return False

    def do_query_procedures(self):
        return [PROC_SEGMENT, PROC_SETUP]

    def do_create_procedure(self, name):
        if name == PROC_SEGMENT:
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run_segment, None)
            procedure.set_image_types("RGB*, GRAY*")
            procedure.set_sensitivity_mask(
                Gimp.ProcedureSensitivityMask.DRAWABLE
                | Gimp.ProcedureSensitivityMask.DRAWABLES
                | Gimp.ProcedureSensitivityMask.NO_DRAWABLES)
            procedure.set_menu_label("Segment _Anything...")
            procedure.add_menu_path("<Image>/Select")
            procedure.set_documentation(
                "Select objects using Meta's Segment Anything models",
                "Runs SAM 2 or SAM 3 on the visible image, lists every region "
                "it finds, and turns the regions you pick into a selection, a "
                "layer mask, a new layer or a channel.",
                name)
            procedure.set_attribution(AUTHOR, AUTHOR, COPYRIGHT_YEAR)
            return procedure

        if name == PROC_SETUP:
            procedure = Gimp.ImageProcedure.new(
                self, name, Gimp.PDBProcType.PLUGIN, run_setup, None)
            procedure.set_image_types("*")
            procedure.set_sensitivity_mask(
                Gimp.ProcedureSensitivityMask.DRAWABLE
                | Gimp.ProcedureSensitivityMask.DRAWABLES
                | Gimp.ProcedureSensitivityMask.NO_DRAWABLES)
            procedure.set_menu_label("_Setup...")
            procedure.add_menu_path("<Image>/Filters/Segment Anything")
            procedure.set_documentation(
                "Install or change the Segment Anything models",
                "Downloads a private Python environment and the SAM 2 or SAM 3 "
                "weights, or removes them again.",
                name)
            procedure.set_attribution(AUTHOR, AUTHOR, COPYRIGHT_YEAR)
            return procedure

        return None


Gimp.main(SegmentAnything.__gtype__, sys.argv)

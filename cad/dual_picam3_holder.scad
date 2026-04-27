// ============================================================
//  Dual Raspberry Pi Camera Module 3 Holder
//  Head-mounted rig, optimized for hand-tracking + scene capture
//
//  Inspired by Printables #396799 (makerbymistake) but restructured:
//    • Both cameras pitched DOWN together (hands live in lower FOV)
//    • Toe angle parameter (+ toe-in, − toe-out)
//    • Adjustable baseline (lens-to-lens spacing)
//    • M2 screw retention for PCB (no friction-fit)
//    • Single M2 mount hole in the bottom tab
//
//  Coordinate convention (world frame):
//    +X = right, +Y = forward (user's gaze), +Z = up
//  Camera local frame (inside camera_plate):
//    +Y = lens-forward, PCB sits in the XZ plane at the back of the plate,
//    +Z = top of PCB (ribbon connector side)
// ============================================================

// ---------------- PRIMARY GEOMETRY (tune freely) ----------------
baseline        = 35;    // mm, lens-center to lens-center
pitch_angle     = -30;   // deg, both cameras pitch DOWN (+ = down, − = up)
toe_in_angle    = 10;    // deg per camera (+ toe-in, − toe-out)

// ---------------- CAMERA MODULE 3 SPEC ----------------
// (Leave alone unless targeting a different board)
pcb_w              = 25;    // PCB width
pcb_h              = 24;    // PCB height
pcb_t              = 4.2;   // PCB overall thickness (used for preview only)
lens_dia           = 8.5;   // lens-barrel clearance diameter
lens_offset_z      = 0;     // lens vertical offset from PCB center (0 = centered)
ribbon_w           = 22;    // ribbon cable exit slot width
ribbon_h           = 2.5;   // ribbon cable thickness clearance

// ---------------- PCB POCKET SHAPE (stepped) ----------------
// The Camera Module 3 is thicker at the ribbon-connector end (FPC
// connector sticks ~5 mm off the PCB back) and thinner everywhere else
// (just 3–3.5 mm of chip height).  We cut a STEPPED pocket so the
// plastic behind the chip zone is maximally thick (= strongest screw
// engagement) while the connector zone gets the extra clearance it
// needs.  The deeper region is anchored to the ribbon end (+Z side).
// Measured off the physical board:
//   • Chips on the back of the PCB stand ~2.5 mm off the PCB surface.
//   • The FPC connector stands ~4.3 mm ABOVE the chips, so total
//     clearance needed in the connector zone = 2.5 + 4.3 = 6.8 mm
//     (we reference every depth to the PCB surface, not to the chips).
//   • The FPC connector body is ~6 mm tall measured from the ribbon
//     edge inward → connector_zone_h = 6.
connector_pocket_d = 6.8;   // pocket depth under the ribbon connector
chip_pocket_d      = 2.5;   // pocket depth under the rest of the PCB
connector_zone_h   = 6.0;   // height of connector zone measured from
                            //   the ribbon edge of the PCB (+Z side)

// ---------------- M2 MOUNTING HOLES ----------------
// Four holes at the four corners of the PCB mounting pattern.
// Anchored at the PCB edge OPPOSITE the ribbon connector (call it
// the "top" edge — in this code's local frame, that's the −Z edge,
// because the ribbon exits at +Z).
//   - "Top" row: two holes hole_top_inset mm in from the far-from-
//     ribbon edge (top-left and top-right corners of the PCB).
//   - "Bottom" row: two more holes hole_dy mm further down, toward
//     the ribbon edge.
hole_dx            = 21;    // horizontal spacing between L and R holes
hole_dy            = 12.5;  // vertical spacing between top and bottom rows
hole_top_inset     = 2.0;   // inset from far-from-ribbon edge to top row

// ---------------- HOLDER STRUCTURE ----------------
wall_t             = 2.5;   // wall around PCB pocket
front_plate_t      = 6.0;   // material in front of the DEEPEST part of the
                            //   pocket (connector zone).  Set by the "don't
                            //   make the whole thing too thick" constraint:
                            //   6 mm keeps the plate slim while still giving
                            //   ~9 mm of material behind the chips (where
                            //   the four M2 screws actually bite), which is
                            //   plenty for M2×6 to M2×8 self-tap screws.
pocket_clear       = 0.4;   // extra depth+width slop in PCB pocket

// ---------------- FASTENERS ----------------
// Self-tap screw holes are STEPPED: a clearance-sized bore for most of
// the length (so the screw shank drops in freely) and a smaller pilot
// diameter only at the tip (where the threads actually bite into
// plastic).  This is the standard practice for self-tap into plastic —
// a straight pilot-through-full-length forces the threads to cut the
// entire depth of the hole, which is what makes screws physically hard
// to drive.  Stepped holes let you sink an M2×8 with a screwdriver in
// one easy turn.
m2_clearance      = 2.5;   // clearance hole for M2 shank (bumped from 2.3
                           //   to 2.5 for FDM print tolerance — a 2.3 mm
                           //   hole prints ~2.1 mm after shrinkage / wall
                           //   squish, and an M2 shank is already ~2.0 mm,
                           //   so there's almost no slop.  2.5 mm gives
                           //   the shank real drop-through clearance.)
m2_pilot          = 1.9;   // pilot diameter for M2 SOCKET HEAD CAP screw
                           //   form-tapping into PLA on a Bambu X1C.
                           //   A cap screw has no cutting flute, so it
                           //   *displaces* plastic instead of cutting it.
                           //   Sizing math: M2 major ⌀ = 2.0 mm, minor
                           //   ⌀ ≈ 1.55 mm.  We want the printed hole to
                           //   land around ⌀1.75 mm so the screw displaces
                           //   ~0.12 mm radially per side — tight enough
                           //   to grip, loose enough to drive with a
                           //   1.5 mm hex key without stripping the hex.
                           //   X1C PLA horizontal holes print ~0.15 mm
                           //   undersize, so CAD value 1.9 → printed 1.75.
m2_thread_engage  = 11.0;  // length (mm) of the PILOT-sized section at
                           //   the FRONT of the bore.  With plate depth
                           //   ≈ 13.2 mm and thread_engage = 11, only the
                           //   back ~2.2 mm is wide clearance (for screw
                           //   alignment lead-in); the remaining 11 mm
                           //   is form-tap pilot.  A 10 mm screw enters
                           //   with ~2 mm of head standoff (PCB in
                           //   pocket) and reaches the pilot zone almost
                           //   immediately, giving ~6 mm of real thread
                           //   engagement — plenty for M2 in PLA.
                           //   WARNING: if you shorten the screw or
                           //   thicken the front plate, recompute this.
                           //   The screw TIP must land inside the pilot
                           //   zone; otherwise the screw spins freely in
                           //   clearance and never bites.
use_self_tap    = true;  // true = stepped clearance+pilot hole (screw
                         //   threads into plastic only at the tip)
                         // false = straight clearance hole all the way
                         //   through (use M2 nuts on the front face)

// ---------------- GOPRO-STYLE FORK MOUNT ----------------
// Two parallel prongs hanging from the back-bottom of the camera body.
//
// Key property: the mount is placed in the body-LOCAL frame and then
// pitched together with the cameras, so the prong's back face stays
// flush against the body's back surface no matter what pitch_angle is
// set to.  Change pitch from -30° to 0° to +45° — the mount follows.
//
// Geometry of each prong (in body-local frame, before pitch rotation):
//   - Back face aligned with body back at Y = −(front_plate_t + pcb_t + pocket_clear)
//     (offset by gopro_flush_offset if you want to nudge it forward/back)
//   - Top anchor sinks UP into the body from its bottom edge
//   - Hinge hangs DOWN below the body by gopro_drop
mount_enable       = true;
gopro_tab_t        = 3.0;   // prong thickness (X dimension)
gopro_tab_gap      = 3.2;   // gap between prongs (fits a 3 mm mating tab)
gopro_tab_r        = 7.5;   // hinge radius (rounded end, total ⌀ = 15 mm)
gopro_hole_dia     = 2.4;   // pin/screw hole ⌀ (M2 = 2.4, standard GoPro = 5.1)
gopro_drop         = 10;    // distance from body bottom edge DOWN to hinge axis
                            //   (body-local Z; must be ≥ gopro_tab_r for the
                            //    hinge to clear the body bottom)
gopro_flush_offset = 0;     // Y offset of prong back face relative to body back
                            //   0 = perfectly flush
                            //   + = prong sits slightly forward of back surface
                            //   − = prong sticks out behind back surface
gopro_anchor_d     = 5;     // depth the top anchor sinks INTO the body
                            //   (along +Y from back surface).  Auto-clamped
                            //   to body depth at render time so the anchor
                            //   never pokes out the front, no matter what.
gopro_anchor_h     = 12;    // height the top anchor rises INTO the body
                            //   (along +Z from bottom edge).  Bigger = more
                            //   vertical grab = stronger mechanical bond.
gopro_embed_eps    = 0.3;   // how far the anchor intrudes PAST the body's
                            //   back and bottom surfaces (≈ one nozzle width).
                            //   Invisible in the final print, but CRITICAL:
                            //   it guarantees strict volumetric overlap so
                            //   CSG produces a single manifold solid instead
                            //   of two bodies touching at a coincident plane
                            //   — which is what makes slicers treat it as a
                            //   surface-only attachment.

// ---------------- RENDERING ----------------
$fn             = 60;
show_pcbs       = false; // true = render translucent PCB+lens previews for sanity

// ============================================================
//  MODULES
// ============================================================

// Total plate depth (Y direction): front_plate_t of material in front
// of the DEEPEST pocket zone (the connector zone), plus the pocket
// itself.  Exposed as a function so other modules can use it.
function plate_total_depth() =
    front_plate_t + connector_pocket_d + pocket_clear;

// Stepped M2 screw hole, drilled along local +Z, starting at z=0.
//   - use_self_tap = true: clearance bore for (length − m2_thread_engage)
//     followed by a pilot-sized section for the final m2_thread_engage
//     mm.  The screw shank passes freely through the wide section; only
//     the threaded tip engages plastic.  This is what makes a self-tap
//     screw easy to drive instead of a wrestling match.
//   - use_self_tap = false: a single clearance-diameter cylinder for
//     the full length (meant to be paired with an M2 nut on the front
//     face — the screw never grips plastic).
// NOTE: the final 0.01 mm overlap between the two cylinders prevents
// a zero-thickness disc from showing up at the step, which would
// otherwise foul some slicers.
module m2_screw_hole(length) {
    if (use_self_tap) {
        clear_len = max(length - m2_thread_engage, 0.01);
        cylinder(d = m2_clearance, h = clear_len + 0.01);
        translate([0, 0, clear_len])
            cylinder(d = m2_pilot,
                     h = length - clear_len + 0.01);
    } else {
        cylinder(d = m2_clearance, h = length);
    }
}

// Solid outer envelope of one camera plate (before any cutouts).
// Local frame: lens axis along +Y, plate front face at Y=0,
// plate back at Y = -total_depth.
module camera_plate_solid() {
    ow = pcb_w + 2*wall_t;
    oh = pcb_h + 2*wall_t;
    td = plate_total_depth();
    translate([-ow/2, -td, -oh/2])
        cube([ow, td, oh]);
}

// All cutouts for one camera: stepped PCB pocket, lens window, M2 screw
// holes, ribbon cable exit slot.
module camera_cutouts() {
    oh     = pcb_h + 2*wall_t;
    td     = plate_total_depth();

    // --- STEPPED PCB POCKET ---
    // Cut #1 (shallow): the chip zone — spans the WHOLE PCB footprint
    // but only chip_pocket_d deep.  This gives us the maximum possible
    // wall thickness behind the chips → strong M2 thread bite.
    translate([-pcb_w/2 - pocket_clear/2,
               -td - 0.01,
               -pcb_h/2 - pocket_clear/2])
        cube([pcb_w + pocket_clear,
              chip_pocket_d + pocket_clear + 0.02,
              pcb_h + pocket_clear]);

    // Cut #2 (deep): the connector zone — only connector_zone_h tall,
    // anchored at the ribbon-connector edge (+Z side of PCB), cut
    // connector_pocket_d deep.  The union of Cut #1 and Cut #2
    // produces a stepped pocket: shallow everywhere, deep just at the
    // top strip where the FPC connector sticks out.
    translate([-pcb_w/2 - pocket_clear/2,
               -td - 0.01,
               pcb_h/2 - connector_zone_h])
        cube([pcb_w + pocket_clear,
              connector_pocket_d + pocket_clear + 0.02,
              connector_zone_h + pocket_clear/2 + 0.01]);

    // --- LENS WINDOW ---
    translate([0, -td - 0.01, lens_offset_z])
        rotate([-90, 0, 0])
            cylinder(d = lens_dia, h = td + 0.02);

    // --- M2 SCREW HOLES (4 corners of the PCB mounting pattern) ---
    // Coordinate reminder: ribbon connector exits at +Z ("bottom" of
    // the PCB in the user's head-mounted orientation, because the
    // cameras are pitched down).  The "top" row of holes — the corners
    // farthest from the ribbon — therefore sit at −Z, inset by
    // hole_top_inset from the far-from-ribbon edge.  The "bottom" row
    // sits hole_dy mm toward the ribbon (+Z) from there.
    //
    // Each hole is drilled from the BACK of the plate and passes ALL
    // THE WAY THROUGH to the front face (td + 0.2 length guarantees
    // breakout on the front).  The screw is inserted from the back,
    // after the PCB drops into the pocket:
    //   • back side (PCB side): clearance bore — the shank drops in
    //     freely, so you can seat the screw by hand before reaching
    //     for the driver.
    //   • front side (last m2_thread_engage mm): pilot bore — threads
    //     bite here and only here.  Exits the front face as a small
    //     ~1.85 mm hole, which is invisible from the back and fine
    //     structurally (it doesn't weaken the front plate materially).
    // Total bore length with the default 6 mm front plate and the
    // connector-zone pocket = plate_total_depth() ≈ 13.2 mm, which
    // handles M2×8 through M2×12 screws.
    top_row_z = -pcb_h/2 + hole_top_inset;   // far from ribbon
    bot_row_z = top_row_z + hole_dy;         // toward ribbon
    hole_positions = [
        [-hole_dx/2, top_row_z], [ hole_dx/2, top_row_z],
        [-hole_dx/2, bot_row_z], [ hole_dx/2, bot_row_z],
    ];
    for (p = hole_positions)
        translate([p[0], -td - 0.1, p[1]])
            rotate([-90, 0, 0])
                m2_screw_hole(td + 0.2);

    // --- RIBBON CABLE EXIT SLOT ---
    // Connector sits on back of PCB at the ribbon edge (+Z); the flex
    // cable must escape upward through the top wall.
    translate([-ribbon_w/2,
               -td - 0.01,
               pcb_h/2 - 0.5])
        cube([ribbon_w,
              td + 0.02,
              wall_t + 2]);
}

// Translucent visual preview of PCB + lens barrel for sanity-checking
// placement / angles.  Not part of the printed geometry.
module pcb_preview() {
    color("green", 0.45)
        translate([-pcb_w/2,
                   -(front_plate_t + pcb_t),
                   -pcb_h/2])
            cube([pcb_w, pcb_t, pcb_h]);
    color("black", 0.6)
        translate([0, -front_plate_t, lens_offset_z])
            rotate([90, 0, 0])
                cylinder(d = 7.5, h = 4);
    // ribbon connector stub (on PCB back)
    color("#222", 0.6)
        translate([-ribbon_w/2,
                   -(front_plate_t + pcb_t + 5),
                   pcb_h/2 - 3])
            cube([ribbon_w, 5, 3]);
}

// Place children at one camera's position and orientation.
// side = -1 → left camera, +1 → right camera.
// Transformation order (inner → outer):
//   1. pitch DOWN around local X (rotate [-pitch,0,0])
//   2. toe around world Z (rotate [0,0,side*toe])
//   3. translate to ±baseline/2 on X
// This keeps "pitch" horizontal-referenced and "toe" vertical-referenced.
module place(side) {
    translate([side * baseline/2, 0, 0])
        rotate([0, 0, side * toe_in_angle])
            rotate([-pitch_angle, 0, 0])
                children();
}

// ---------------- GOPRO FORK MOUNT (pitch-aware, back-flush) ----------------
// Total outer width of the fork along X (both prongs + gap)
function gopro_total_w() = 2*gopro_tab_t + gopro_tab_gap;

// Body reference coordinates in the body-LOCAL frame (i.e. BEFORE the
// pitch rotation is applied).  These describe the un-pitched camera
// body's back surface and bottom edge, so the mount can be attached to
// them regardless of what pitch_angle is set to.
function body_back_y()   = -plate_total_depth();
function body_bottom_z() = -(pcb_h + 2*wall_t)/2;

// Body Y-depth (distance from back surface to lens face, un-pitched).
// Used to CLAMP the anchor depth so it can never poke out the front.
function body_depth_y() = plate_total_depth();

// Effective anchor depth: clamped to just under the body depth so the
// anchor always lives strictly inside the body's Y extent.
function anchor_d_eff() =
    min(gopro_anchor_d, body_depth_y() - 0.2);

// One visible prong — built in the body-local frame.  Back face is
// parallel to the body's back face and offset by gopro_flush_offset.
// The top anchor is pushed PAST the body's back and bottom planes by
// gopro_embed_eps so the union has strict volumetric overlap (no
// coincident surfaces ⇒ clean single manifold in STL export).
module gopro_prong(side) {
    // Prong occupies X ∈ [x_lo, x_lo + gopro_tab_t]
    x_lo = side > 0
        ?  gopro_tab_gap/2
        : -gopro_tab_gap/2 - gopro_tab_t;

    // Back face of the prong, in body-local Y (0 = flush with body back)
    prong_back_y = body_back_y() + gopro_flush_offset;
    bot_z        = body_bottom_z();

    translate([x_lo, 0, 0])
        hull() {
            // Hinge end — rounded cylinder hanging below body.
            // Back tangent of the cylinder aligns with prong_back_y.
            translate([0,
                       prong_back_y + gopro_tab_r,
                       bot_z - gopro_drop])
                rotate([0, 90, 0])
                    cylinder(r = gopro_tab_r, h = gopro_tab_t, $fn = 80);

            // Top anchor — a block that sinks forward (+Y) and upward
            // (+Z) INTO the body.  Its back face and bottom face each
            // protrude past the body's corresponding surfaces by
            // gopro_embed_eps so CSG produces a volumetric union, not
            // a surface-coincident touch.
            translate([0,
                       prong_back_y - gopro_embed_eps,
                       bot_z - gopro_embed_eps])
                cube([gopro_tab_t,
                      anchor_d_eff() + gopro_embed_eps,
                      gopro_anchor_h  + gopro_embed_eps]);
        }
}

// Structural back buttress — a wide, thick slab of material that sits
// INSIDE the body, spanning between the two prongs and extending up
// along the back wall.  It carries the load from the hinge prongs up
// into the body mass, turning the mount into an integral part of the
// body spine rather than a bottom appendage.
//   - X: spans the full fork width (plus a bit of extra grip)
//   - Y: starts at the back surface, extends forward by anchor_d_eff()
//   - Z: starts at the body bottom, rises up by gopro_anchor_h
// Like the prong anchor, it protrudes past the back/bottom by
// gopro_embed_eps to keep the union strictly volumetric.
module gopro_back_buttress() {
    w = gopro_total_w() + 2;     // slightly wider than the fork
    prong_back_y = body_back_y() + gopro_flush_offset;
    bot_z        = body_bottom_z();

    translate([-w/2,
               prong_back_y - gopro_embed_eps,
               bot_z - gopro_embed_eps])
        cube([w,
              anchor_d_eff() + gopro_embed_eps,
              gopro_anchor_h  + gopro_embed_eps]);
}

// Both prongs + buttress together — pitched with the body so the
// whole mount stays rigidly attached to the back surface at any
// pitch_angle.  The buttress is what makes the mount a structural
// part of the body, not just a glued-on appendage.
module gopro_prongs() {
    rotate([-pitch_angle, 0, 0]) {
        gopro_prong(-1);
        gopro_prong(+1);
        gopro_back_buttress();
    }
}

// Pin / screw hole through both prongs along the X axis.  Also pitched
// with the body so it stays coaxial with the hinge cylinders.
module gopro_pin_hole() {
    prong_back_y = body_back_y() + gopro_flush_offset;
    bot_z        = body_bottom_z();
    w            = gopro_total_w();
    rotate([-pitch_angle, 0, 0])
        translate([-w/2 - 1,
                   prong_back_y + gopro_tab_r,
                   bot_z - gopro_drop])
            rotate([0, 90, 0])
                cylinder(d = gopro_hole_dia, h = w + 2, $fn = 40);
}

// ============================================================
//  ASSEMBLY
// ============================================================

module dual_holder() {
    difference() {
        union() {
            // Camera bridge — hull of ONLY the two camera plates.
            // This keeps the bridge compact and leaves the camera fronts
            // exposed.  The fork is NOT included in this hull, so it
            // stays external (matching the reference design).
            hull() {
                place(-1) camera_plate_solid();
                place(+1) camera_plate_solid();
            }
            // External fork — two independent prongs whose anchors
            // poke up into the camera bridge for a solid union.  No
            // slot subtraction needed: the gap between the prongs is
            // an inherent feature of drawing them as separate pieces.
            if (mount_enable) gopro_prongs();
        }

        // Per-camera cutouts (pocket, lens, screws, ribbon slot)
        place(-1) camera_cutouts();
        place(+1) camera_cutouts();

        // Pin hole through both prongs
        if (mount_enable) gopro_pin_hole();
    }

    if (show_pcbs) {
        place(-1) pcb_preview();
        place(+1) pcb_preview();
    }
}

dual_holder();

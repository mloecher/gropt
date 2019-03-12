#############
Constraints
#############


Below are a list of constraints and documentation about their usage and input formats.

Where possible, the units are all Tesla, meters, and seconds (though occasionally ms will come up for helper functions where it might make more sense)

Most constraints are subject to a 1% cushion to help convergence of the optimization, so many constriants might be 99% of their actual value.

Gradient Strength
==================

Maximum gradient strength :math:`(G_{max})` is defined in :math:`\dfrac{T}{s}`

This constraint simply clips the gradient waveforms so that all values must be in the range :math:`(-G_{max}, G_{max})`

Slew Rate
=================

Maximum slew rate :math:`(SR_{max})` is defined in :math:`\dfrac{T}{(m \cdot s)}`

This constraint constrains :math:`\dfrac{dG}{dt}` so that all values must be in the range :math:`(-SR_{max}, SR_{max})`

.. _ref-moment-constraints:

Moments
==================

Moments are controlled with an array of doubles of length N_momentsx7, where N_moments is the number of moment constraints to be applied.   Any number of constraints (N_moments) can be added in this array.

The 7 options in a constraint are (sequentially):

- Axis of constraint
    0, 1, or 2 based on which axis to apply the constraint on, when more than one axis of gradients are being computed.

    *Not currently implemented*

- Moment order
    What order of moment the constraint is. i.e. 0 = :math:`M_{0}`, 1 = :math:`M_{1}`, 2 = :math:`M_{2}`.
    
    *Technically any number should work, though nothing higher than 2 is routinely tested.*

- t=0 offset
    This field gives the offset in **seconds** between the first point in the waveform, and the actual t=0 time for the moment integrals.
    
    e.g. If there is a 2 ms rf pulse before the waveform that is not included in the waveform, but you would still like to calculate moments from the excitation, this argument could = 1e-3 (1 ms)

- Start time
    Defines the beginning of the range of the waveform that the moment should be constrained over

    -1 means the very beginning of the waveform (default behavior, to calculate for the entire waveform)

    (0, 1) gives a time in **seconds** relative to the start of the waveform (not the previous argument reference)

    >1 is cast to an integer and gives the index of the point to start the calculation

    *Not currently implemented*

- End time
    Defines the end of the range of the waveform that the moment should be constrained over

    -1 means the very end of the waveform (default behavior, to calculate for the entire waveform with -1 above)

    (0, 1) gives a time in **seconds** relative to the start of the waveform (not the previous argument reference)

    >1 is cast to an integer and gives the index of the point to end the calculation

    *Not currently implemented*

- Desired Moment
    The desired moment, with units :math:`\dfrac{mT \cdot ms^{N+1}}{m}`, where :math:`N` is the moment order (see second argument)

- Moment Tolerance
    How far off the waveform can be from the desired moment, in the same units.

    *The optimization will very frequently use exactly all of this tolerance*

.. _ref-eddy-constraints:

Eddy Currents
==================

Eddy current constraints are controlled with an array of doubles of length N_eddyx4, where N_eddy is the number of eddy current constraints to be applied.   Any number of constraints (N_eddy) can be added in this array.

The 4 options in a constraint are (sequentially):

- Time constant of constraint
    The time constant in milliseconds of the constraint to be applied

- Target eddy current magnitude
    The target value of the constraint (0 for nulling, but can be set to other values).  the units here are basically arbitrary.

- Tolerance
    The tolerance in the target value to be used.  The units are somewhat arbitrary, but you can start around 1.0e-4

- Mode
    0.0 to constrain the instantaneous eddy currents at the end of the waveform.  1.0 to constrain the sum of eddy current fields across the whole waveform.
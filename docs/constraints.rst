Constraints
************

Below are a list of constraints and documentation about their usage and input formats.

Where possible, the units are all Tesla, meters, and seconds (though occasionally ms will come up for helper functions where it might make more sense)

Most constraints are subject to a 1% cushion to help convergence of the optimization, so many constriants might be 99% of their actual value.

Gradient Strength
------------------

Maximum gradient strength :math:`(G_{max})` is defined in :math:`\dfrac{T}{s}`

This constraint simply clips the gradient waveforms so that all values must be in the range :math:`(-G_{max}, G_{max})`

Slew Rate
------------------

Maximum slew rate :math:`(SR_{max})` is defined in :math:`\dfrac{T}{(m \cdot s)}`

This constraint constrains :math:`\dfrac{dG}{dt}` so that all values must be in the range :math:`(-SR_{max}, SR_{max})`
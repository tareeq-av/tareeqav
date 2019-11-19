
TareeqAV
======

TareeqAV is a  platform that provides practical hands-on knowledge and real-world experience that is affordable; yet comparable to the real challenging scenarios of self-driving, to help engineers master the techniques and skills required to be competitive in this very exciting and fast moving industry.

Implementations
------

TareeqAV provides independent implementations in *Python* and *C++*.  *Python* implementations are meant to be introductory to the software modules that make up a self-driving car, while the *C++* implementation attemps to closely mimic a production quality code base.

#### Python Implementation


The *Python* implementation comprises two independent stages with increasing complexity.


###### Stage 1

**Currently Under Development**

Stage 1 uses out-of-the-box open source implementations found in Github repositories of state-of-the-art research papers in __perception__, __planning__, and control.

We use each model found in the Github repo as is and for the task for which it was created withgout any attempt at merging them into a multi-task network.

###### Stage 2

**Development Beginning in January 2020**

Stage 2 will combine the networks of *Stage 1* into a __Single Multi Task Network__.


Please view the [README](./python/README.md) of the python directory for more details.

#### C++ Implementation

**Development Beginning in April 2020**

The *C++* implemenation begins where Python's *Stage 2* ends.  We will re-write *Stage 2* into *C++* and begin developing the runtime framework from the ground up.


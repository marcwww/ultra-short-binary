# ultra-short-binary
This code is completed referring to the paper, "USB: Ultrashort Binary Descriptor for Fast Visual Matching and Retrieval"(http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6832500&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel7%2F83%2F4358840%2F06832500.pdf%3Farnumber%3D6832500)

In the current version, low efficiency is still a problem:
it is really fast in feature matching, but it does not work well in feature extraction, and the bottlenecks are the two methods:
surf();(provided by opencv) as well as usb_extraction();, further work could be done around editting these two methods.

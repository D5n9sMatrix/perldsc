#!/usr/bin/perl

=head1 NAME
 
Code - notes and coding rules
 
=head1 DESCRIPTION
 
The documentation for using wurst lives in F<wurst.pod>. This
is a description of internals, conventions and how to change the code.
 
=head1 Layout
 
=over
 
=item Philosophy
 
Have a look at some example files. There is a standard layout for
making functions stand out.  Please do not use your own layout,
even if it is prettier.  The aim is consistency and
predictability.
 
=item Tabs
 
There are B<no> tab characters. Use spaces. Use four spaces to
indent in functions and loops.
 
=item Indentation
 
Indentation is almost pure Kernighan and Ritchie with four
spaces.
 
=item Line size
 
Try to make code readable in a window of 80 columns or printed
page of 80 columns.  This may mean shortening your line or
breaking it in a place you think makes it more readable for
others.
 
=item Version control
 
All the files have lines like this:
 
  #if !defined (lint) && !defined (DONT_SEE_RCS)
      static const char *rcsid =
      "$Id: coding.pod,v 1.1 2007/09/28 16:57:19 mmundry Exp $";
  #endif /* !defined (lint) && !defined (DONT_SEE_RCS) */
 
These come after the #include directives.
 
=item Function headings
 
Begin every function as in the examples. Do not use your own
style, even if you think yours is better. Every function begins:
 
    /* ---------------- foobar  -----------------------------------
     * This function takes an argument and makes coffee.
     */
    static int
    foobar (int zot) 
    {
        ....
 
To list the properties:
 
=over
 
=item *
the comment style at the start should be respected,
including the little stars and the empty column.
 
=item *
function type on its own line
 
=item *
the opening brace is on its own line
 
=back
 
=item Punctuation
 
There should usually be a space after punctuation, for exactly
the same reasons as in English (helping
readability). Commas and semi-colons definitely want to be
followed by a space.
 
Do not write
 
    for (a=0;a<lim;a++){
 
Do
 
    for (a = 0; a < lim; a++) {
 
If the line is going to go beyone 79 spaces, delicately delete
spaces while trying to keep readability.
 
=item typedef
 
A typedef is useful for complicated structures.  In many other
places it obscures the meaning. In wurst, typedefs should never
go into function prototypes. If they do, then other files cannot
easily include the function prototype. Consider a very
complicated structure of type S<C<struct tree>>. In a header you
may write
 
 struct tree;
 int function (struct tree *t);
 
Then anyone can include it without needing the definition of
S<C<struct tree>>.
 
If you write
 
 int function (treestruct *t);
 
Then nobody can include it without seeing the C<typedef>.
 
=item include files
 
=over 
 
=item *
 
Do not include files from other include files.
 
=item *
 
Split the description of the internal structures from the
interface. Call the file with the internal structures, C<blah.h>
and the one with the interface, C<blah_i.h>.
 
=back
 
=item boolean
 
Do not add your own type with a name like "boolean" for
true/false.  If you are talking about errors, use
C<EXIT_SUCCESS>/C<EXIT_FAILURE> since these are ANSI standard. If
you have a yes/no situation, then there is a header file with a
simple enumerated type in F<yesno.h>.  The problem with your own
boolean is that it gets redefined by different people, TRUE and
FALSE often appear in unrelated header files and the most recent
C standard has its own idea about boolean types.
 
=back
 
=head1 INTERNALS
 
=head2 Declarations
 
=over
 
=item Ordering
 
The order in which you declare variables should make no
difference. On the compaq, at least within structures, the
compiler makes noise unless you order objects from largest to
smallest.  This avoids padding to get best alignment.
 
=item Function declarations
 
If a function can possibly be made static, it should be.  Symbols
should be no more visible than is necessary.  There are also
optimisation advantages (the compiler can assume a function is
local and generate only an inlined version).
 
If you have a static function you must
 
=over
 
=item *
 
Not declare it, then define it and use it. Instead,
 
=item *
 
You should define it once, before it is used.
 
=back
 
In other words, do
 
 static int function (int arg) { ... }
 ...
 b = function (a);
 
Rather than
 
 static int function (int arg);
 ...
 static int function (int arg) { ... }
 ...
 b = function (a);
 
Obviously this rule has to be broken if there is mutual
recursion.
 
This rule exists for consistency (it is used in every file) and
to follow the principle that we should write things the minimum
number of times and lastly, to ensure that whenever possible, a
compiler is given all the information about a static function by
the time it is used.
 
=back
 
=head2 Return values and errors
 
There are two aspects to this.
 
=over
 
=item Error messages
 
Generally, the C functions do not print out too much information
in case of error. They must, however, return an error code to the
interpreter. The C functions should use C<err_printf()> to pass
on the result of C<strerror()>/C<perror()>. Typically, this means
the lower level C code will write something like C<no such file
or directory>, but it is up to the interpreter/script to decide
whether or not to go on.
 
=item Return codes
 
Most of the C code returns C<EXIT_SUCCESS> if it is happy and
C<EXIT_FAILURE> (as defined in F<stdlib.h>). This leads to a
problem. C<EXIT_FAILURE> may be defined as C<1>, but the perl
interpreter usually takes C<1> as indicating success. The
convention in this code is that
 
=over
 
=item *
 
The C code returns C<EXIT_SUCCESS>/C<EXIT_FAILURE> as it
pleases. The interface code (F<Wurst.xs>) handles swapping the
sense when necessary. So, if C code returns an error, the
interface code may do C<XSRETURN_UNDEF>.
 
=back
 
=back
 
=head2 Conventions and internal routines
 
Wurst code may not call any of the following directly.
 
=over
 
=item printf()
 
=item fprintf()
 
At least not to stderr.
 
=item malloc()
 
=back
 
There are some general areas to think about.
 
=over
 
=item printing
 
Functions should not print to stdout or stderr
directly. Everything should go via C<err_printf()> or
C<mprintf()> or C<mfprintf()>.
 
If you have to print, you code must include
 
#include "mprintf.h"
 
=over
 
=item mprintf()
 
Takes exactly the same arguments as C<printf()>.
 
=item mfprint()
 
Exactly the same as C<fprintf()>.
 
=item err_printf()
 
This is a replacement for S<C<fprintf(stderr, "blah")>>, but
it is not a direct wrapper. The convention is
 
 #include <stdio.h>
 #include "mprintf.h"
    ...
 int
 foo_bar (...)
 {
     const char *this_sub = "foo_bar";
     ...
     if (error)
         err_printf (this_sub, "blah %s message\n", args);
 }
 
This will result in a printing to something like stderr a
string like
 
 Function foo_bar: blah args messsage
 
=item mputchar()
 
=item mfputc()
 
=item mputs()
 
=item mfputs()
 
=item mperror()
 
All these functions with names like mxxx(), behave exactly as xxx().
 
=back
 
=item Passing strings to interpreter
 
If you have a static string to pass back to the interpreter,
declare the function as
 
 char *
 some_func (..)
 
and all will be well. This. however, is the exception. Normally,
there is a string to be printed out and it has been dynamically
allocated. After the interpreter has finished with the string, it
should be free()'d. There are two ways to go about this.
 
=over
 
=item * perls SV mechanism
 
=item * our internal scr_xxx() mechanism
 
=back
 
Both the approaches are fine, but perhaps the first (perl's
mechanism) should be preferred from now on.
 
Imagine your function allocates space for a string, writes
something into it and you want the interpreter to be able to
print it out.
 
In the C code, declare your function returning char * like
 
    char *
    my_func (...)
    {
        char *x = E_MALLOC (sizeof (x[0] * length);
        strcpy (x, "something");
        return x;
    }
 
Then, in the file, F<something.xs>, use an interface like
 
SV *
my_func (a)
        int a;
    CODE:
        {
            char *s = my_func(...);
            RETVAL = newSVpv (s, 0);
            free (s);
        }
    OUTPUT:
        RETVAL
 
This will copy the string into a C<SV> which the interpreter
knows how to deal with. We can immediately free() the pointer
from our function.
 
The second method is to use the scratch space routines,
C<scr_printf()> like this.
 
    #include "scratch.h"
 
    char *s;
    scr_reset();
    s = scr_printf ("blah %s\n", foo);
    s = scr_printf ("more text\n");
    return s;
 
The first call C<scr_reset()> is necessary to tell our scratch
space to free any old strings. Subsequent calls append the
strings.
 
=item Memory allocation and malloc
 
Do not call C<malloc()>. Instead, call the macro
C<E_MALLOC()>. This takes exactly the same arguments as
C<malloc()>, but expands to print out the file and line number
if C<malloc()> fails.
 
It is a conscious design decision that we do not do any error
recovery there. If we run out of memory, we are cactus.
 
=back
 
=head2 Memory allocation philosophy and responsibility
 
Most of the time, memory is malloc()'d and free()'d at the
same level of code. There is a major exception to
this. Objects which are passed back to the interpreter rely on
the interpreter's garbage collection. You must declare and
appropriate routine like foo_blah_DESTROY(). This will be
called by the interpreter when the reference count for the
object goes to zero.  This routine must clean up any allocated
space.
 
This approach works well for objects like score
matrices. These are allocated once. Although their contents
may be manipulated, they do not move. Other objects, such as
sequence arrays may grow or shrink and move about as
determined by calls to E_REALLOC(). This means we have an
extra level of redirection and the interpreter is actually
given a pointer to a pointer to a sequence array.
 
 
 
=head2 Score matrix storage
 
The score matrix for two objects of size B<M> and B<N> is of
size, S<B<(M + 2) * (N + 2)>>. The reason is that it is easier
to do special treatment of ends and end gaps this way.  This
means one has to be careful when coding around the score
matrices.
 
=head2 Sequences and sequence arrays
 
In an early version, functions operated on arrays of
sequences. This is being phased out. Some functions may still
be able to handle arrays of sequences, mainly for getting
sequences in FASTA format.
 
=head2 Structures and special rules
 
=over
 
=item struct seq
 
=over
 
=item size
 
These items hold a sequence. The size element holds the number of
residues, but the string is allocated for S<(n + 1)> so we can
really treat it as a string with a null terminator.
 
=item format
 
Sequence strings can either be conventional, one-letter amino
acid names or in 'thomas' format. The strings have an enumerated
type to say what state they are in. If your code depends on some
format, then call the appropriate function to convert the string
from one style to the other. For example, score functions will
often force strings to 'THOMAS' format. Before printing, strings
will usually be converted to 'PRINTABLE' format.
 
=back
 
=item struct coord
 
The coordinate structure not only holds coordinates, it holds a
corresponding sequence in a C<struct seq> containe within. The
number of elements in the coord and seq structures must be the
same.
 
Within the coord structure, code may B<not> assume that the
secondary strucure (sec_s) is complete, nor the phi angles
(phi). If these are NULL pointers, it means they have not been
read up, calculated or filled out.  The C<coord_DESTROY> routine
has a look to see if these are NULL pointers and calls C<free()>
only if they are non-NULL.
 
=item struct prob_vec
 
Probability vectors are complicated because
 
=over
 
=item *
 
They can be in the simple array or a compact format
 
=item *
 
They may be normalised in two different ways.
 
=item *
 
The number of sites and probability vectors may vary.
 
=back
 
 
In more detail
 
=over
 
=item prob_vec data storage
 
=over
 
=item expanded
 
In expanded form, the data is in a dynamically allocated
2D array. In mship[a][b], the slower changing index, C<a> is the
site in the protein. The faster, C<b>, is the class index.
 
=item compressed
 
In the C<mship[a][b]> example, most of the elements are near
zero. In the compressed form, C<cmpct_n-E<gt>cmpct_n> is an array
where each element tells us how many non-zero probabilities are
associated with site C<a>. For site C<i>, C<cmpct_n[i]> says how
many probabilities are stored in C<prob_vec-E<gt>cmpct_prob>. For
each of those probabilities, there is a corresponding entry in
C<prob_vec-E<gt>cmpct_ndx> which is the number (index) of the
associated class. The number of elements in C<cmpct_prob> will be
the same as in C<cmpct_ndx>.
 
=item protocol
 
If there is no compressed data, C<prob_vec-E<gt>cmpct_n> must point
to C<NULL>. If C<prob_vec-E<gt>cmpct_n> is non-null, the pointer
will be assumed valid. If there is no expanded data,
C<prob_vec-E<gt>mship> must point to C<NULL>.
 
 
=back
 
=item prob_vec normalisation
 
In terms of probabilities, the vector should be normalised so
that all the entries sum to 1.0. When comparing objects, we want
to treat the probability vectors as vectors of unit length, so we
can take the dot product in order to see the similarity. We also
have to allow for not knowing the state of our vectors. These
three situations are coded for by setting
C<prob_vec-E<gt>norm_type> to 
 
=over
 
=item *
 
C<PVEC_TRUE_PROB>
 
=item *
 
C<PVEC_UNIT_VEC>
 
=item *
 
C<PVEC_CRAP>
 
=back
 
=item prob_vec number of sites / vectors
 
Describe the number of sites.
 
=back
 
=back
 
=head1 ADDING WURST
 
To add a function to wurst, you have to add something to
 
=over
 
=item Wurst/Wurst.xs
 
This is the C/perl interface
 
=item Wurst/Wurst.pm
 
This advertises the symbols which the scripts can use.
 
=item The C file
 
The actual code to be called.
 
=item pod/wurst.pod
 
This is the documentation.
 
=back
 
In more detail.
If you want to add a function, B<do_stuff> which acts on
something of type B<thing_struct>, then do at least the
following.
 
=head2 C code
 
In F<do_stuff.c>, or wherever, define the function like
 
  int
  do_stuff (struct thing_struct *thing) {
     ....
  }
 
In F<do_stuff.h>, prototype the function interface like
 
  int do_stuff (struct thing_struct *thing);
 
Don't define the structure here. It is not necessary for the
perl interface to see structure internals.
 
=head2 XS code
 
In Wurst/Wurst.xs, add
 
  #include "do_stuff.h"
 
and a typedef
 
  typedef struct thing_struct Thing_struct;
 
The capitalisation is not a joke. Perl likes it.
Finally, still in F<Wurst/Wurst.xs>, add the function
interface,
 
  int
  do_stuff (x)
      Thing_struct *x;
 
=head2 .pm code
 
Go to F<Wurst.pm>. Find the @EXPORT section and add
B<do_stuff>.
 
=head2 .pod documentation
 
Go to F<pod/wurst.pod> and add a description of the new
function under the heading, B<FUNCTIONS>
 
If the function returns some data type back to the
interpreter, add a mention of that data type as well.
 
=head1 DEBUGGING
 
=head2 General debugging
 
There are two kinds of debugging:
 
=over
 
=item perl debugging
 
This is not so interesting. Use the perl debugger.
 
=item debugging the C code
 
This is more fun and discussed below.
 
=back
 
Wurst is coded as a perl extension, but perl does not have
debugging symbols. During development, we will typically
compile the extension with C<-g>. The problem is that you
cannot set breakpoints since at program load time, the perl
extension (a dynamically loaded library) is not there and the
debugger does not see a symbol table.
 
This is not a terrible problem. The trick is to send a little
signal after the program has started, but before the
breakpoint. There are two ways to do this.
 
=over
 
=item signal from perl
 
Somewhere early in the perl script, before entering the function
where you want a breakpoint, insert
 
    $SIG{TRAP} = 'IGNORE';
    kill 'TRAP', $$;
 
You could also set C<$SIG{TRAP}> to point to some function of
your own.
 
These lines say that we will ignore TRAP signals.  Then the code
sends a TRAP to itself.  The debugger notices the signal and
gives you a command prompt. Now, the dynamic library is loaded
and breakpoints can be set. This works with either B<gdb> or
B<dbx>.
 
The nice aspect of this approach is that when one is not
debugging, the script runs fine. The signal is sent, but
ignored.
 
=item signal from C
 
Sometimes one knows pretty well which function is going to be
a problem, and where one wants to look closely. We can use the
same mechanism by calling C<kill()> from the C code. The
easiest way to do this is to find the file of interest.  At
the top of the file,
 
    #include "dbg.h"
 
In the subroutine of interest, insert a line
 
    breaker();
 
This little function just gets out the current process ID and
sends it a trap'able signal. The debugger will give you a command
prompt inside the breaker() function which can be quickly stepped
out of.
 
=back
 
=head2 Memory debugging
 
This is a challenge. One wants to work with a debugging
C<malloc()> library, which looks for leaks and so on. When
building perl, it offers you the use of its own C<malloc()>
library or the system one. It is probably useful to build perl
with the system library.
The
memory checking under C<dbx> on Solaris works fine. Getting an
add-on library like B<electric fence> to work under solaris is
a bore. Wild use of the LD_PRELOAD environment variable
results in a program which gobbles up all available
memory. Under linux, it seems to be no problem to link
B<electric fence> in.
 
We assume that perl itself is free of leaks and we only want to
instrument our library.
 
=head1 Rules
 
Please follow these rules when writing wurst code.
 
 
=head2 Function interfaces
 
Keep the definition of the interface minimal and separate from
the definition of the innards. If you have a C file call
F<foo.c>, then
 
=over
 
=item *
 
Put the public interface in F<foo_i.h>
 
=item *
 
Put the innnards in a file like F<foo.h>
 
=back
 
For example
 
You may have some code like
 
    struct cow {
        int legs;
        float weight;
    };
 
    int
    print_cow ( struct cow * daisy)
    {
        ....
    }
 
 
Now there are a few files which have to know the innards of a
S<C<struct cow>> and others which will want to call
print_cow().
In this case, create a file F<cow.h> which defines the
structure:
 
    struct cow {
        int legs;
        float weight;
    };
 
and in a separate file, F<cow_i.h>,
 
    struct cow;
    int print_cow (struct cow *c);
 
Note:
 
=over
 
=item *
 
Callers of print_cow() do not need to see the innards of the
cow struct.
 
=item *
 
You should B<not> use a typedef in the function defintion or
prototype. If the code said
 
  struct cow {...}.
  typedef struct cow Cow;
  typedef struct cow *CowPtr;
 
then the prototype for print_cow() would be
 
  int print_cow (CowPtr);
 
but then B<every> caller would need a definition of CowPtr.
Please do B<not> do that.
 
=back
 
=head2 Text and messages
 
=over
 
=item mfprintf.c
 
Use the wrappers in F<mprintf.c>.
These may not do anything fancy now, but they allow us to wrap
our output at a later stage. For example, a Tcl extension is
not allowed to read or write to S<F<stdout> / F<stderr>>. In
that case, the wrappers can do whatever is necessary to get
printf(), fprintf() functions.
 
=item File Opening
 
It is so frequent that we call fopen() with an error message,
that this is now in the wrapper, mfopen() in F<fio.c>. Use
 
 #include "fio.h"
 FILE * mfopen (const char *fname, const char *mode, const char *s)
 
where B<fname> and B<mode> are as for fopen() and B<s> is a
string which will go into error messages. Typically, this will
be the name of the caller.
 
=item File Reading and Caching
 
We have a hook to ask the OS not not to cache a file after
reading it for opening. Use the function C<file_no_cache(fp)> on
the FILE pointer, C<fp>. This somtimes makes a surprising difference in
performance. When reading a library of structures or profiles,
wurst may read almost 10000 files sequentially. The OS tries to
cache each one, although they will not be read again. The damage
is that it pushes useful pages such as other people's programs
out of the cache.  The function C<file_no_cache()> is a wrapper
around a posix function which may not be present, so the wrapper
checks appropriate C<#define>d values and is a no-op if the posix
functions do not seem to be present. A typical usage would be,
    if ((fp = mfopen (fname, "r", this_sub)) == NULL)
        return NULL;
 
    {
        int tmp;
        const char *s = "Turning off caching for %s:\n\"%s\"\n";
        if ((tmp = file_no_cache(fp)) != 0)
            err_printf (this_sub, s, fname, strerror (tmp));
    }
 
RETURN value is zero on success or C<errno> on failure.
 
=item Reading lines from files
 
Very often, we want to read a line from a file, hopping over
blank lines and lines that begin with a hash (#).
Do not write another function to do this.
 
 #include "misc.h"
 get_nline (FILE *fp, char *buf, int *nr_line, size_t maxbuf)
 
Reads a line from fp. into buffer buf whose size is given by
maxbuf.
nr_line points to a line counter. get_nline() will increment
the variable on every line read.
 
The routine throws away anything on a line after the hash (#)
character, so you can have inline comments.
 
=back
 
=head2 amino acids
 
There are a couple of common conversions for amino acid names
and types. We may go from one to three letter codes and,
internally, from Thomas to standard one letter codes. All
functions for this kind of thing live in F<amino_a.c>.
 
=head2 memory allocation
 
=over
 
=item standard malloc()
 
All memory allocation must be done through the macros
B<E_MALLOC()> and B<E_REALLOC()> which live in
F<e_malloc.c>. They take the standard arguments and return the
standard results, but, on failure, print out a line saying
where the error occurred, how much memory was requested and
then they die. This is generally good behaviour. Fancier code
could attempt error recovery, but usually when malloc() fails,
it means there was a code error.
 
=item matrices
 
All two dimensional matrices should be allocated via the
routines which have their roots in Thomas' code (which is
basically the outline given in Numerical Recipes).
 
=back
 
=head2 Function naming
 
Some days, I call my functions C<print_thing()> and
C<print_foo()>. On others, I use C<foo_print()> and
C<thing_print()>.  Neither is better than the other. Here is
the rule...
 
Functions should be named C<thing_print()>.
 
Note, I am aware of the disadvantage that you cannot see all
the things you can print or copy or whatever.
 
=cut
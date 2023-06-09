=head1 NAME
 
Tk::Dialog - Perl/Tk Dialog widget
 
=for pm Tk/Dialog.pm
 
=for category Popups and Dialogs
 
=head1 SYNOPSIS
 
    require Tk::Dialog;
 
    $DialogRef = $widget->Dialog(
        -title          => $title,
        -text           => $text,
        -bitmap         => $bitmap,
        -default_button => $default_button,
        -buttons        => [@button_labels],
    );
 
    $selected = $DialogRef->Show(?-global?);
 
=head1 DESCRIPTION
 
This is an OO implementation of `tk_dialog'.  First, create all your B<Dialog>
objects during program initialization.  When it's time to use a dialog,
invoke the C<Show> method on a dialog object; the method then displays the
dialog, waits for a button to be invoked, and returns the text label of the
selected button.
 
A Dialog object essentially consists of two subwidgets: a Label widget for
the bitmap and a Label wigdet for the text of the dialog.  If required, you
can invoke the `configure' method to change any characteristic of these
subwidgets.
 
Because a Dialog object is a Toplevel widget all the 'composite' base class
methods are available to you.
 
Advertised widgets:  bitmap, message.
 
=over 4
 
=item 1)
 
Call the constructor to create the dialog object, which in turn returns
a blessed reference to the new composite widget:
 
    require Tk::Dialog;
 
    $DialogRef = $widget->Dialog(
        -title          => $title,
        -text           => $text,
        -bitmap         => $bitmap,
        -default_button => $default_button,
        -buttons        => [@button_labels],
    );
 
=over 4
 
=item * mw
 
a widget reference, usually the result of a C<MainWindow-E<gt>new> call.
 
=item * title
 
Title to display in the dialog's decorative frame.
 
=item * text
 
Message to display in the dialog widget.
 
=item * bitmap
 
Bitmap to display in the dialog.
 
=item * default_button
 
Text label of the button that is to display the
default ring (''signifies no default button).
 
=item * button_labels
 
A reference to a list of one or more strings to
display in buttons across the bottom of the dialog.
 
=back
 
=item 2)
 
Invoke the C<Show> method on a dialog object
 
    $button_label = $DialogRef->Show;
 
This returns the text label of the selected button.
 
(Note:  you can request a global grab by passing the string C<-global>
to the C<Show> method.)
 
=back
 
=head1 SEE ALSO
 
Tk::DialogBox
 
=head1 KEYWORDS
 
window, dialog, dialogbox
 
=head1 AUTHOR
 
Stephen O. Lidie, Lehigh University Computing Center.  94/12/27
lusol@Lehigh.EDU (based on John Stoffel's idea).
 
=cut
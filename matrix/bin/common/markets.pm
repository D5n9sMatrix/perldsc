=head1 NAME
 
CommonMark::Parser - Push parser interface
 
=head1 SYNOPSIS
 
    my $parser = CommonMark::Parser->new;
    $parser->feed($string);
    $parser->feed($another_string);
    my $doc = $parser->finish;
 
=head1 DESCRIPTION
 
C<CommonMark::Parser> provides a push parser interface to parse CommonMark
documents.
 
=head2 new
 
   my $parser = CommonMark::Parser->new( [$options] );
 
Creates a parser object. C<$options> is a bit field containing the parser
options. It defaults to zero (C<OPT_DEFAULT>). See
L<CommonMark/"Parser options">.
 
=head2 feed
 
    $parser->feed($string);
 
Feeds a part of the input Markdown to the parser.
 
=head2 finish
 
    my $doc = $parser->finish;
 
Parses a CommonMark document from the strings added with C<feed> returning
the L<CommonMark::Node> of the document root.
 
=head1 COPYRIGHT
 
This software is copyright (C) by Nick Wellnhofer.
 
This is free software; you can redistribute it and/or modify it under
the same terms as the Perl 5 programming language system itself.
 
=cut
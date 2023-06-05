package Array::To::Moose;
 
# Copyright (c) Stanford University. June 6th, 2010.
# All rights reserved.
# Author: Sam Brain <samb@stanford.edu>
# This library is free software; you can redistribute it and/or modify
# it under the same terms as Perl itself, either Perl version 5.8.8 or,
# at your option, any later version of Perl 5 you may have available.
#
 
use 5.008008;
use strict;
use warnings;
 
require Exporter;
use base qw( Exporter );
 
our %EXPORT_TAGS = (
    'ALL'     => [ qw( array_to_moose
                       throw_nonunique_keys throw_multiple_rows
                       set_class_ind set_key_ind                 ) ],
    'TESTING' => [ qw( _check_descriptor _check_subobj
                       _check_ref_attribs _check_non_ref_attribs ) ],
);
 
our @EXPORT_OK = ( @{ $EXPORT_TAGS{'ALL'} }, @{ $EXPORT_TAGS{'TESTING'} } );
 
our @EXPORT = qw( array_to_moose 
 
);
 
use version; our $VERSION = qv('0.0.9');
 
# BEGIN { $Exporter::Verbose=1 };
 
#BEGIN { print "Got Array::To:Moose Module\n" }
 
package Params::Validate::Array;
package Array::GroupBy;
use Carp;
use Data::Dumper;
 
$Carp::Verbose = 1;
 
$Data::Dumper::Terse  = 1;
$Data::Dumper::Indent = 1;
 
# strings for "key => ..." and "class => ..." indicators
my ($KEY, $CLASS);
 
BEGIN { $KEY = 'key' ; $CLASS = 'class' }
 
# throw error if a HashRef[] key found to be non-unique
my $throw_nonunique_keys;
 
# throw  if there are multiple candidate rows for an attribute
# which is a single object, "isa => 'MyObject'"
my $throw_multiple_rows;
 
############################################
# Set the indicators for "key => ..." and "class => ..."
# If there is no arg, reset them back to the default 'key' and 'class'
############################################
sub set_key_ind {
  croak "set_key_ind('$_[0]') not a legal identifier"
    if defined $_[0] and $_[0] !~ /^\w+$/;
 
  $KEY = defined $_[0] ? $_[0] : 'key';
}
 
############################################
sub set_class_ind {
  croak "set_class_ind('$_[0]') not a legal identifier"
    if defined $_[0] and $_[0] !~ /^\w+$/;
 
  $CLASS = defined $_[0] ? $_[0] : 'class';
}
 
########################################
# throw error if non-unique keys in a HashRef['] is causing already-constructed
# Moose objects to be overwritten
# throw_nonunique_keys() to set, throw_nonunique_keys(0) to unset
########################################
sub throw_nonunique_keys { $throw_nonunique_keys = defined $_[0] ? $_[0] : 1 }
 
########################################
# throw error if a single object attribute has multiple data rows
# throw_multiple_rows() to set throw_multiple_rows(0) to unset
########################################
sub throw_multiple_rows { $throw_multiple_rows = defined $_[0] ? $_[0] : 1 }
 
##########
# Usage
#   my $moose_object_ref = array_to_moose( data => $array_ref,
#                                          desc => { ... },
#                                        );
############################################
sub array_to_moose {
  my $data = shift;
  croak "'data => ...' isn't a 2D array (AoA)"
    unless ref($data->[0]);
 
  croak 'empty descriptor'
    unless keys %$data;
 
  #print "data ", Dumper($data), "\ndesc ", Dumper($desc);
 
 
  my $result = [];   # returned result is either an array or a hash of objects
 
  # extract column of possible hash key
  my $keycol;
 
  if (exists $data->{$KEY}) {
    $keycol = $data->{$KEY};
 
    $result = {};         # returning a hashref
 
  }
 
  # _check_descriptor returns:
  # $class,       the class of the object
  # $attribs,     a hashref (attrib => column_number) of "simple" attributes
  #               (column numbers only)
  # $ref_attribs, a hashref of attribute/column number values for
  #               non-simple attributes, currently limited to "ArrayRef[`a]",
  #               where `a is e.g 'Str', etc (i.e. `a is not a class)
  # $sub_desc,    a hashref of sub-objects.
  #               the keys are the attrib. names, the values the
  #               descriptors of the next level down
 
  my ($class, $attribs, $ref_attribs, $sub_obj_desc) =
            _check_descriptor($data);
 
  #print "data ", Dumper($data), "\nattrib = ", Dumper($attribs),
  #      "\nargs = ", Dumper([ values %$attribs ]);
 
  #print "\$ref_attribs ", Dumper($ref_attribs); exit;
 
  my $iter = igroup_by(
                data    => $data,
                compare => \&str_row_equal,
                args    => [ values %$attribs ],
  );
 
  while (my $subset = $iter->()) {
 
    #print "subset: ", Dumper($subset), "\n";
 
    #print "before 1: attrib ", Dumper($attribs), "\ndata ", Dumper($subset);
 
    # change attribs from col numbers to values:
    # from:  { name => 1,           sex => 2,      ... }
    # to     { name => 'Smith, J.', sex => 'male', ... }
    my %attribs = map { $_ => $subset->[0]->[$attribs->{$_}] } keys %$attribs;
    
 
    # print "after 1: attrib ", Dumper(\%attribs), "\n";
 
    # add the 'simple ArrayRef' sub-objects
    # (there should really be only one of these - test for it?)
    while (my($attr_name, $col) = each %$ref_attribs) {
      my @col = map { $_->[$col] } @$subset;
      $attribs{$attr_name} = \@col;
 
      # ... or ...
      #$attribs{$attr_name} = [ map { $_->[$col] } @$subset ];
    }
 
    # print "after 2: attrib ", Dumper(\%attribs), "\n";
 
    # sub-objects - recursive call to array_to_moose()
    while( my($attr_name, $desc) = each %$sub_obj_desc) {
 
      my $type = $class->meta->find_attribute_by_name($attr_name)->type_constraint
        or croak "Moose attribute '$attr_name' has no type";
 
      #print "'$attr_name' has type '$type'";
 
      my $sub_obj = array_to_moose( data => $subset,
                                    desc => $desc,
                                  );
 
      $sub_obj = _check_subobj($class, $attr_name, $type, $sub_obj);
 
      #print "type $type\n";
 
      $attribs{$attr_name} = $sub_obj;
    }
 
    # print "after 2: attrib ", Dumper(\%attribs), "\n";
 
    my $obj;
    eval { $obj = $class->meta->new_object(%attribs) };
    croak "Can't make a new '$class' object:\n$@\n"
          if $@;
 
    if (defined $keycol) {
      my $key_name = $subset->[0]->[$keycol];
 
      # optionally croak if we are overwriting an existing hash entry
       croak "Non-unique key '$key_name' in '", $data->{$CLASS}, "' class"
        if exists $result->{$key_name} and $throw_nonunique_keys;
 
      $result->{$key_name} = $obj;
    } else {
      push @{$result}, $obj;
    }
  }
  return $result;
}
 
############################################
# Usage: my ($class, $attribs, $ref_attribs, $sub_desc)
#                  = _check_descriptor($data, $desc)
#
# Check the correctness of the descriptor hashref, $desc.
#
# Checks of descriptor $desc include:
# 1. "class => 'MyClass'" line exists, and that class "MyClass" has
#                         been defined
# 2. for "attrib => N" 
#     or "key    => N" lines, N, the column number, is an integer, and that
#                      the column numbers is within limits of the data
# 3. For "attrib => [N]", (note square brackets), N, the columnn number,
#                         is within limits of the data
#
# Returns:
# $class,      the class name,
# $attribs,    hashref (name => column_index) of "simple" attributes
# $ref_attribs hashref (name => column_index) of attribs which are
#               ArrayRef[']s of simple types (i.e. not a Class)
#               (HashRef[']s not implemented)
# $sub_desc    hashref (name => desc) of sub-object descriptors
############################################
sub _check_descriptor {
  my ($data, $desc) = @_;
 
  # remove from production!
  croak "_check_descriptor() needs two arguments"
    unless @_ == 2;
 
  my $class = $desc->{$CLASS}
    or croak "No class descriptor '$CLASS => ...' in descriptor:\n",
       Dumper($desc);
 
  my $meta;
 
  # see other example of getting meta in Moose::Manual::???
  eval{ $meta = $class->meta };
  croak "Class '$class' not defined: $@"
    if $@;
 
  my $ncols = @{ $data->[0] };
 
  # separate out simple (i.e. non-reference) attributes, reference
  # attributes, and sub-objects
  my ($attrib, $ref_attrib, $sub_desc);
 
  while ( my ($name, $value) =  each %$desc) {
 
    # check lines which have 'simple' column numbers ( attrib or key => N)
    unless (ref($value) or $name eq $CLASS) {
 
      my $msg = "attribute '$name => $value'";
 
      croak "$msg must be a (non-negative) integer"
        unless $value =~ /^\d+$/;
 
      croak "$msg greater than # cols in the data ($ncols)"
        if $value > $ncols - 1;
    }
 
    # check to see if there are attributes called 'class' or 'key'
    if ($name eq $CLASS or $name eq $KEY) {
      croak "The '$class' object has an attribute called '$name'"
        if $meta->find_attribute_by_name($name);
 
      next;
    }
 
    croak "Attribute '$name' not in '$class' object"
      unless $meta->find_attribute_by_name($name);
 
    if ((my $ref = ref($value)) eq 'HASH') {
      $sub_desc->{$name} = $value;
 
    } elsif ($ref eq 'ARRAY') {
      # descr entry looks like, e.g.:
      #   attrib => [6],
      #
      # ( or attrib => [key => 6, value => 7],  in future... ?)
 
      croak "attribute must be of form, e.g.: '$name => [N], "
            . "where N is a single integer'"
          unless @$value == 1;
 
      my $msg = "attribute '$name => [ " . $value->[0] . " ]'. '" .
                  $value->[0] . "'";
 
      croak "$msg must be a (non-negative) integer"
        unless $value->[0]  =~ /^\d+$/;
 
      croak "$msg greater than # cols in the data ($ncols)"
        if $value->[0] > $ncols - 1;
 
      $ref_attrib->{$name} = $value->[0];
 
    } elsif ($ref) {
      croak "attribute '$name' can't be a '$ref' reference";
 
    } else {
      # "simple" attribute
      $attrib->{$name} = $value;
    }
  }
 
 
  # check ref- and ...
  _check_ref_attribs($class, $ref_attrib)
    if $ref_attrib;
 
  # ... non-ref attributes from the descriptor against the Moose object
  _check_non_ref_attribs($class, $attrib)
    if $attrib;
 
  croak "no attributes with column numbers in descriptor:\n", Dumper($desc)
    unless $attrib and %$attrib;
 
  return ($class, $attrib, $ref_attrib, $sub_desc);
}
 
########################################
# Usage: $sub_obj = _check_subobj($class, $attr_name, $type, $sub_obj);
#
# $class        is the name of the current class
# $attr_name    is the name of the attribute in the descriptor, e.g.
#               MyObjs => { ... } (used only diagnostic messages)
# $type         is the expected Moose type of the sub-object
#               i.e. 'HashRef[MyObj]', 'ArrayRef[MyObj]', or 'MyObj'
# $sub_obj_ref  Reference to the data (just returned from a recursive call to
#               array_to_moose() ) to be stored in the sub-object,
#               i.e. isa => 'HashRef[MyObj]', isa => 'ArrayRef[MyObj]',
#               or isa => 'MyObj'
#
#
# Checks that the data in $sub_obj_ref agrees with the type of the object to
# contain it
# if $type is a ref to an object (isa => 'MyObj'), _check_subobj() converts
# $sub_obj_ref from an arrayref to sub-object to ref to a subobj
# (see notes in code below)
#
# Throws error is it finds a type mis-match
########################################
sub _check_subobj {
  my ($class, $attr_name, $type, $sub_obj) = @_;
 
  # for now...
  croak "_check_subobj() should have 4 args" unless @_ == 4;
 
  #my $type = $class->meta->find_attribute_by_name($attr_name)->type_constraint
  #  or croak "Moose class '$class' attribute '$attr_name' has no type";
 
  if ( $type =~ /^HashRef\[([^]]*)\]/ ) {
 
    #print "subobj is of type ", ref($sub_obj), "\n";
    #print "subobj ", Dumper($sub_obj);
 
    croak "Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor produced an object "
          . "of type '" . ref($sub_obj) . "'\n"
      if ref($sub_obj) ne 'HASH';
 
    #print "\$1 '$1', value: ", ref( ( values %{$sub_obj} )[0] ), "\n";
 
    croak("Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor produced an object "
          . "of type 'HashRef[" . ref( ( values %{$sub_obj} )[0] )
          . "]'\n")
      if ref( ( values %{$sub_obj} )[0] ) ne $1;
 
  } elsif ( $type =~ /^ArrayRef\[([^]]*)\]/ ) {
 
    croak "Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor produced an object "
          . "of type '" . ref($sub_obj) . "'\n"
      if ref($sub_obj) ne 'ARRAY';
 
    croak "Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor produced an object "
          . "of type 'ArrayRef[" . ref( $sub_obj->[0] ) . "]'\n"
      if ref( $sub_obj->[0] ) ne $1;
 
  } else {
 
    # not isa => 'ArrayRef[MyObj]' or 'HashRef[MyObj]' but isa => 'MyObj',
    # *but* since array_to_moose() can return only a hash- or arrayref of Moose
    # objects, $sub_obj will be an arrayref of Moose objects, which we convert to a
    # ref to an object
 
    croak "Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor generated a '"
          . ref($sub_obj)
          . "' object and not the expected ARRAY"
      unless ref $sub_obj eq 'ARRAY';
 
    # optionally give error if we got more than one row
    croak "Expected a single '$type' object, but got ",
        scalar @$sub_obj, " of them"
      if @$sub_obj != 1 and $throw_multiple_rows;
 
    # convert from arrayref of objects to ref to object
    $sub_obj = $sub_obj->[0];
 
    # print "\$sub_obj type is ", ref($sub_obj), "\n";
 
    croak "Moose attribute '$attr_name' has type '$type' "
          . "but your descriptor produced an object "
          . "of type '" . ref( $sub_obj ) . "'"
      unless ref( $sub_obj ) eq $type;
  }
  return $sub_obj;
}
 
{
 
  # The Moose type hierarchy (from Moose::Manual::Types) is:
  # Any
  # Item
  #     Bool
  #     Maybe[`a]
  #     Undef
  #     Defined
  #         Value
  #             Str
  #                 Num
  #                     Int
  #                 ClassName
  #                 RoleName
  #         Ref
  #             ScalarRef[`a]
  #             ArrayRef[`a]
  #             HashRef[`a]
  #             CodeRef
  #             RegexpRef
  #             GlobRef
  #                 FileHandle
  #             Object
 
  # So the test for 
 
  my %simple_types;
 
  BEGIN
  {
    %simple_types = map { $_ => 1 }
      qw ( Any Item Bool Undef Defined Value Str Num Int __ANON__ );
  }
 
########################################
# Usage:
#   _check_ref_attribs($class, $ref_attribs);
# Checks that "reference" attributes from the descriptor (e.g., attr => [N])
# are ArrayRef[]'s of simple attributes in the Moose object
# (e.g., isa => ArrayRef['Str'])
# Throws an exception if check fails
#
# where:
#   $class is the current Moose class
#   $ref_attribs an hashref of Moose attributes which are "ref
#   attributes", e.g., " has 'hobbies' (isa => 'ArrayRef[Str]'); "
#
########################################
sub _check_ref_attribs {
  my ($class, $ref_attribs) = @_;
 
  my $meta = $class->meta
    or croak "No meta for class '$class'?";
 
  foreach my $attrib ( keys %{ $ref_attribs } ) {
    my $msg = "Moose class '$class' ref attrib '$attrib'";
 
    my $constraint = $meta->find_attribute_by_name($attrib)->type_constraint
      or croak "$msg has no type constraint";
 
    #print "_check_ref_attribs(): $attrib $constraint\n";
 
    if ($constraint =~ /^ArrayRef\[([^]]*)\]/ ) {
 
      croak "$msg has bad type '$constraint' ('$1' is not a simple type)"
        unless $simple_types{$1};
 
      return;
    }
    croak "$msg must be an ArrayRef[`a] and not a '$constraint'";
  }
}
 
 
########################################
# Usage:
#   _check_non_ref_attribs($class, $non_ref_attribs);
# Checks that non-ref attributes from the descriptor (e.g., attr => N)
# are indeed simple attributes in the Moose object (e.g., isa => 'Str')
# Throws an exception if check fails
#
#
# where:
#   $class is the current Moose class
#   $non_ref_attribs an hashref of Moose attributes which are 
#   non-reference, or "simple" attributes like 'Str', 'Int', etc.
#   The key is the attribute name, the value the type
#
########################################
sub _check_non_ref_attribs {
  my ($class, $attribs) = @_;
 
  my $meta = $class->meta
    or croak "No meta for class '$class'?";
 
  foreach my $attrib ( keys %{ $attribs } ) {
    my $msg = "Moose class '$class', attrib '$attrib'";
 
    my $constraint = $meta->find_attribute_by_name($attrib)->type_constraint
      or croak "$msg has no type (isa => ...)";
 
    #print "_check_non_ref_attribs(): $attrib '$constraint'\n";
 
    # kludge for Maybe[`]
    $constraint =~ /^Maybe\[([^]]+)\]/;
    $constraint = $1 if $1;
 
    #print " after: $attrib '$constraint'\n";
 
    next if $simple_types{$constraint};
 
    $msg = "$msg has type '$constraint', but your descriptor had '$attrib => "
         . $attribs->{$attrib} . "'.";
 
    $msg .= " (Did you forget the '[]' brackets?)"
      if $constraint =~ /^ArrayRef/;
       
    croak $msg;
  }
}
       
} # end of local block
 
 
1;
 
__END__

 
# TODO
#
# test for non-square data array?
#
# - allow argument "compare => sub {...}" in array_to_moose() call to
# allow a user-defined row-comparison routine to be passed to
# Array::GroupBy::igroup_by()
#
# - make it Mouse-compatible? (All meta->... stuff would break?)
 
##### SUBROUTINE INDEX #####
#                          #
#   gen by index_subs.pl   #
#   on 24 Apr 2014 21:11   #
#                          #
############################
 
 
####### Packages ###########
 
# Array::To::Moose ......................... 1
#   array_to_moose ......................... 2
#   set_class_ind .......................... 2
#   set_key_ind ............................ 2
#   throw_multiple_rows .................... 2
#   throw_nonunique_keys ................... 2
#   _check_descriptor ...................... 4
#   _check_non_ref_attribs ................. 9
#   _check_ref_attribs ..................... 8
#   _check_subobj .......................... 6
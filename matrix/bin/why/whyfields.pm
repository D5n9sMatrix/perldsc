=encoding utf-8
 
=head1 NAME
 
whyfields - or Modern use of fields.pm and C<%FIELDS>.
 
=head1 DESCRIPTION
 
Here I try to explain why L<fields> is useful B<still>
and also propose alternative way of use C<%FIELDS>.
 
=head1 fields.pm - old story.
 
L<fields> allows you to extend C<use strict> style typo-check
to B<fields> (or slots, properties, attributes, instance variables, member variables.. as you like).
 
    use strict;
    package Cat {
 
       use fields qw/name birth_year/; # field declaration
 
       sub new {
          my Cat $self = fields::new(shift); #  my TYPE $var.
          $self->{name}       = shift; # Checked statically!
          $self->{birth_year} = shift  # Checked statically!
             // $self->_this_year;
          $self;
       }
 
       sub age {
          my Cat $self = shift;
          return $self->_this_year
                    - $self->{birth_year}; # Checked statically!
       }
 
       sub _this_year {
          (localtime)[5] + 1900;
       }
    };
 
    my @cats = map {Cat->new($_, 2010)} qw/Tuxie Petunia Daisy/;
 
    foreach my Cat $cat (@cats) {
 
       print $cat->{name}, ": ", $cat->age, "\n";
 
       # print $cat->{namae}, "\n"; # Compile-time error!
    }
 
Above program defines a class C<Cat>,
with members C<{name}> and C<{birth_year}>.
It also defines constructor C<new> and method C<age>,
which computes cat's age from birth_year.
 
In above program, variables C<$self>, C<$cat> are
declared with type annotation C<Cat> like C<my Cat $self>
so that detect typos about members at compile time (eg. C<perl -wc>).
 
Since this typo-check can be applied as soon as program became
syntactically correct, you can check it very early stage of development
(even when you have no unit tests!). And if you integrate this check
to editor's file-save-hook (using flycheck and/or L<App::perlminlint>),
you can detect typos just after every file savings.
 
=head2 Why most people do not use fields.pm?
 
Today, C<use strict> is well known best practice for perl programming.
If C<fields> is useful too, why is it rarely used?
 
Here is my guess list.
 
=over 4
 
=item Type annotation becomes too long for real world apps
 
In real world application, most classnames are very long
like C<MyProject::SomeModule::SomeClass>. But you wouldn't love to write
following:
 
   my MyProject::SomeModule::SomeClass $obj = ...;
 
I saw some codes uses C<__PACKAGE__> like following, but it is still long.
 
  my __PACKAGE__ $obj = ...;
 
=item fields.pm doesn't generate accessors/constructor
 
In OOP, encapsulation (of internal) is important topic.
Directly accessing C<< $cat->{birth_year} >> outside from C<Cat>
is simply violation of OO philosophy.
 
So we need accessors and constructor.
But C<fields.pm> does nothing about them. So you need to
use other accessor generator like L<Class::Accessor>, anyway.
 
=item fiels.pm was limited to single inheritance.
 
C<fields.pm> was introduced when perl was 5.005.
At that time, it was actually based on ARRAY,
so it was single inheritance only.
 
Then after release of perl5.009, C<fields::new> returns real HASH.
But above restriction kept in C<fields.pm> for backward compatibility.
 
=back
 
=head1 A few tips you should know about fields.
 
So, you might think L<Class::Accessor::Fast> or L<Moo>...
is final answer. But wait, are they check fields typos for you at compile time?
I don't think so. (If I'm wrong, please let me know).
Also, use of accessor in internal code can slowdown your code
(remember perl's sub call is not so fast than simple hash access).
 
In my humble opinion, compile time typo checking is still strong point of
perl5 over other LL like ruby, python and php.
I hope more perl mongers cares about this.
 
Anyway, I want to introduce some facts about C<fields>
so that you could stop worrying and start using C<%FIELDS>.
 
=head2 C<fields> works even for unblessed HASH!
 
Since typo check by C<fields> and type annotation (C<my Cat $cat>) is
executed at compile time, actual value in the variable is not limited
to instances of annotated class (C<Cat>). Even unblessed HASH can be used.
 
For example, you can check L<PSGI> C<$env> statically
like following (this is a shorthand version of L<MOP4Import::PSGIEnv>):
 
 
   use strict;
   use 5.012;
   {
      package Env;
      use fields qw/REQUEST_METHOD psgi.version/; # and so on...
   };
 
   return sub {
      (my Env $env) = @_;
 
      if ($env->{REQUEST_METHOD} eq 'GET') { # Checked!
         return [200, header(), ["You used 'GET'"]];
      }
      elsif ($env->{REQUEST_METHOD} eq 'POST') { # Checked!
         return [200, header(), ["You used 'POST'"]];
      }
      else {
         return [200, header()
                , ["Unsupported method $_\n", "psgi.version="
                   , join(" ", $env->{'psgi.version'})]]; # Checked too!
      }
   };
 
   sub header {
      ["Content-type", "text/plain"]
   }
 
=head2 constant sub can be used as shorthand(type-alias) for C<my TYPE> slot.
X<TYPENAME-alias>
 
In fact, you can shorten type annotation using
B<constant sub> which returns actual long class name.
So, you can rewrite following:
 
   # OLD:
   my MyProject::SomeModule::Purchase $obj = ...;
 
   # NEW:
   sub Purchase () {'MyProject::SomeModule::Purchase'}
   ...
   my Purchase $obj = ...;
 
 
Some of you may feel above acceptable to write.
 
And bonus point of having classname sub is that you can use it
to abstract-out classname for future overriding in subclass.
 
  # OLD: Class name is hardcoded.
  sub create_purchase {
    my $self = shift;
    ...
    my Purchase $obj = MyProject::SomeModule::Purchase->new(...);
    ...
    return $obj
  }
 
  # NEW: Class name becomes overridable.
  sub create_purchase {
    my $self = shift;
    ...
    my Purchase $obj = $self->Purchase->new(...);
    ...
    return $obj
  }
 
Note: Unlike C<< my Purchase $obj >>,
C<< $self->Purchase >> is method invocation.
So later doesn't become constant and subclass can override it.
 
 
=head2 values of C<%FIELDS> can be anything (at least for now).
 
Actually L<fields> is an abstraction interface of perl's core interface
C<%FIELDS>. For example:
 
  package Cat;
  use fields qw/name birth_year/;
 
Above code briefly does following:
 
  package Cat;
  BEGIN {
    our %FIELDS;
    $FIELDS{name} = 1;
    $FIELDS{birth_year} = 2;
  }
 
After the C<BEGIN {...}> block is successfully executed,
perl continues compiling rest of the code
with knowledge of package C<Cat> and its C<%FIELDS> relation.
 
When C<my> variable declaration has type alias like C<my Cat $cat>,
it is marked with the type.
And then compiler find field access like C<< $cat->{name} >>,
it looks up C<%CAT::FIELDS> and checks if C<$CAT::FIELDS{name}> exists.
If it exists, field access is valid. If it doesn't,
you will get compilation error like following:
 
  No such class field "namae" in variable $cat of type main::Cat
 
 
Interestingly, above story tells nothing about the value of C<$CAT::FIELDS{name}>.
Actually, the values are only used in fields.pm to achieve
single inheritance restriction. Perl's core itself
does not care its content.
 
I think this means we are able to write our own alternative to C<fields.pm>
using C<%FIELDS>, B<at our own risks>.
So, let's start experimenting!
 
=head1 (Proposed) Modern use of fields and strict.
 
Based on the above discussion, here I propose alternative style use
of fields (actually C<%FIELDS>) to obtain more typo checking at compile time
like use strict. In short:
 
=over 4
 
=item * Limited use of direct field access for method implementations.
 
=item * my MY $obj is enough short.
 
=item * generate accessors from C<%FIELDS> and have generic constructor.
 
=back
 
=head2 Private use of direct field access.
X<direct-field-access>
 
First, we should divide and conquer our problem.
In this case, I want to divide it over the boundary of encapsulation.
That is B<"public interfaces"> and B<"private implementations">.
 
To isolate the private implementations from the public interfaces,
we must use some kind of accessor (and constructor) subs anyway.
 
Having such public interfaces,
use of direct HASH member access in each method implementations
no more harms public interface. It is totally internal matter.
 
 
   # Direct HASH access from user code is evil.
   package main;
   my $foo = Foo->new(...);
   print $foo->{width} * $foo->{height}; # The evils.
 
   # In implementation code, direct HASH access has no problem.
   package Foo {
     use fields qw/width height/;
     sub area {
       my Foo $self = shift;
       $self->{width} * $self->{height};
     }
   };
   # Users of `Foo->area` doesn't care
   # whether it is direct access or accessor call.
   print $foo->area;
 
For example, suppose you once provided an accessor method C<width()>
and later want to add a read-logging for it,
you can rename the field C<width> to C<_width>
and change accessor C<width()> to provide the logging.
This change is totally invisible to class users.
 
Also note when you change the field C<width> to C<_width>,
perl will tell you which line should be changed compile-time!
 
=head2 C<my MY $obj>
X<MY> X<MY-alias>
 
Second, I propose shorthand name C<MY> as
default L<type alias|/TYPENAME-alias>. Then every method argument declaration
starts like C<(my MY $self, ...) = @_>. I hope this will be short enough
style change to adapt, especially if you are already familiar with C<my>
(it's only 3 chars increase!).
To define this alias, just write C<sub MY () {__PACKAGE__}> at the beginning of
your package.
 
  package MyApp::Model::Company::LongLongProductName {
    sub MY () {__PACKAGE__}; # This!
    use fields qw/price/;
 
    sub add_price {
      (my MY $self, my $val) = @_;
      $self->{price} += $val;
      $self
    }
  };
 
I propose this because IMHO spending time for a shorthand type name of
merely C<$self> doesn't make sense and
having compile-time checking for C<$self> is far more important than it.
(Do you really want to change variable name of the C<$self>
for every classes? Life is short, isn't it?)
 
Of course if you already reached a good type-alias naming for your current code,
just use it. This tip is about time-saving.
 
=head2 accessor generator + generic constructor
X<accessor-generator> X<configure> X<Tk-style-configure>
 
Third, let's generate accessors from C<%FIELDS>. As noted before,
compile-time field checking doesn't rely on the value of
C<$FIELDS{$field_name}>. So we can put our nice field specifications
(like I<readonly>, I<default value>, I<type> ...) there, at own risks;-).
 
Also, to make our accessors really useful, we should also provide
a base constructor which is designed to work well with the accessors.
Because accessors are about object states(fields/properties/attributes...)
and constructor defines initial state of the object.
 
In this document, I will show you a tiny getter generator,
a base updater method (named C<configure>) and a base constructor.
It is rooted in L<"PerlE<sol>Tk"|Tk::options> and tcl/tk widget API.
Such object is used like followings:
 
   # key => value list
   my $obj = Foo->new(width => 8, height => 3);
 
   # HASH is ok too.
   $obj = Foo->new({width => 8, height => 3});
 
Once object is created, its properties(fields) are fetched by name.
 
   print $obj->width * $obj->height; # => 24
 
To update its properties, invoke C<configure> with key-value pair list.
 
   $obj->configure(height => 3, width => 3);
 
   print $obj->width * $obj->height; # => 9
 
=over 4
 
=item * Getters are generated only for fields which have name starting
with C<[A-Za-z]>. Other fields are private.
 
=item * In this style, I generate only getters from fields declaration.
 
=item * If you want to have complex getter, change the field name
starting with '_' and write down your getter by hand.
 
   sub dbh {
     (my Foo $foo) = @_;
     $foo->{_dbh} //= do {
        DBI->connect($foo->{user}, $foo->{password}, ...);
     };
   }
 
=item * For updating, I define general purpose updater C<configure> in base class.
And it eventually calls C<onconfigure_...> hooks if it exists.
 
   sub onconfigure_file {
     (my Foo $foo, my $fn) = @_;
     $foo->{string} = read_file($fn);
   }
 
=item * To set default values,
define C<after_new> in your class.
(There would be better name though :-<).
 
   sub after_new {
     (my Foo $foo) = @_;
     $foo->SUPER::after_new();
     $foo->{name}       //= "(A cat not yet named)";
     $foo->{birth_year} //= $foo->default_birth_year;
   }
   sub default_birth_year {
     _this_year();
   }
 
=back
 
Here is a sample implementation of above behavior.
 
Note: below doesn't care about subclassing.
To achieve it, you must merge C<%FIELDS> from base class.
Easiest way to do it is using L<base>. 
Alternatively, you may implement some kind of equiv of
L<MOP4Import::Declare/declare_as_base>.
 
    use strict;
    use 5.009;
    package MyProject::Object { sub MY () {__PACKAGE__}
       use Carp;
       use fields qw//; # Note. No fields could cause a problem.
       sub new {
         my MY $self = fields::new(shift);
         $self->configure(@_) if @_;
         $self->after_new;
         $self
       }
       sub after_new {}
     
       sub configure {
          my MY $self = shift;
          my (@task);
          my $fields = _fields_hash($self);
          my @params = @_ == 1 && ref $_[0] eq 'HASH' ? %{$_[0]} : @_;
          while (my ($name, $value) = splice @params, 0, 2) {
            unless (defined $name) {
              croak "Undefined key for configure";
            }
            unless ($name =~ /^[A-Za-z]\w*$/) {
              croak "Invalid key for configure $name";
            }
            if (my $sub = $self->can("onconfigure_$name")) {
              push @task, [$sub, $value];
            } elsif (not exists $fields->{$name}) {
              confess "Unknown configure key: $name";
            } else {
              $self->{$name} = $value;
            }
          }
          $$_[0]->($self, $$_[1]) for @task;
          $self;
       }
     
       sub _fields_hash {
         my ($obj) = @_;
         my $sym = _globref($obj, 'FIELDS');
         unless (*{$sym}{HASH}) {
           *$sym = {};
         }
         *{$sym}{HASH};
       }
       sub _globref {
         my ($thing, $name) = @_;
         my $class = ref $thing || $thing;
         no strict 'refs';
         \*{join("::", $class, defined $name ? $name : ())};
       }
     
       # Poorman's MOP4Import::Declare.
       sub import {
          my ($myPack, @decls) = @_;
          my $callpack = caller;
          *{_globref($callpack, 'ISA')} = [$myPack];
          foreach my $decl (@decls) {
             my ($pragma, @args) = @$decl;
             $myPack->can("declare_$pragma")->($myPack, $callpack, @args);
          }
       }
     
       sub declare_fields {
         my ($myPack, $callpack, @names) = @_;
         my $fields = _fields_hash($callpack);
         foreach my $name (@names) {
           $fields->{$name} = 1; # or something more informative.
           *{_globref($callpack, $name)} = sub { $_[0]->{$name} };
         }
       }
    };
    1;
 
 
Here is user code of above base class.
 
    package MyProject::Product; sub MY () {__PACKAGE__}
    use MyProject::Object [fields => qw/name price/];
 
    1;
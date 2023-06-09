=head1 NAME
 
CommonMark::Iterator - Iterate CommonMark nodes
 
=head1 SYNOPSIS
 
    use CommonMark qw(:node :event);
 
    my $iter = $doc->iterator;
 
    while (my ($ev_type, $node) = $iter->next) {
        my $node_type = $node->get_type;
 
        if ($node_type == NODE_PARAGRAPH) {
            if ($ev_type == EVENT_ENTER) {
                print("<p>");
            }
            else {
                print("</p>\n");
            }
        }
        elsif ($node_type == NODE_TEXT) {
            print($node->get_literal);
        }
    }
 
=head1 DESCRIPTION
 
C<CommonMark::Iterator> provides a convenient way to walk through the nodes
in a parse tree.
 
=head2 Construction
 
   my $iterator = $node->iterator;
 
Creates an iterator from a node. C<$node> is the root node of the iterator.
 
=head2 next
 
    my $ev_type = $iterator->next;
    my ($ev_type, $node) = $iterator->next;
 
The contents of the iterator are initially undefined. After the first and
each subsequent call to C<next>, the iterator holds a new event type and a
new current node. In scalar context, C<next> returns the new event type.
In list context, it returns a 2-element list consisting of the new event
type and the new current node.
 
Event types are:
 
    CommonMark::EVENT_DONE
    CommonMark::EVENT_ENTER
    CommonMark::EVENT_EXIT
 
Event types can be imported from L<CommonMark> with tag C<event>.
 
    use CommonMark qw(:event);
 
The iterator starts by visiting the root node. Every visited node C<V>
generates the following sequence of events.
 
=over
 
=item *
 
Enter the node. The event type is C<CommonMark::EVENT_ENTER> and the current
node is set to the entered node C<V>.
 
=item *
 
Visit all children of the node C<V> from first to last applying this sequence
of events recursively.
 
=item *
 
Except for leaf nodes, exit the node. The event type is
C<CommonMark::EVENT_EXIT> and the current node is set to the original node
C<V>.
 
=back
 
After the root node was exited, the event type is set to
C<CommonMark::EVENT_DONE> and the current node to C<undef>. In scalar
context, C<next> returns C<CommonMark::EVENT_DONE>. In list context, it
returns the empty list.
 
For leaf nodes, no exit events are generated. Leaf nodes comprise the node
types that never have children:
 
    CommonMark::NODE_HTML
    CommonMark::NODE_HRULE
    CommonMark::NODE_CODE_BLOCK
    CommonMark::NODE_TEXT
    CommonMark::NODE_SOFTBREAK
    CommonMark::NODE_LINEBREAK
    CommonMark::NODE_CODE
    CommonMark::NODE_INLINE_HTML
 
For other node types, an exit event is generated even if the node has no
children.
 
It is safe to modify nodes after an exit event, or an enter event for leaf
nodes. Otherwise, changes to the tree structure can result in undefined
behavior.
 
=head2 Accessors
 
    my $node    = $iter->get_node;
    my $ev_type = $iter->get_event_type;
    my $node    = $iter->get_root;
 
These accessors return the current node, the current event type, and the
root node.
 
=head1 COPYRIGHT
 
This software is copyright (C) by Nick Wellnhofer.
 
This is free software; you can redistribute it and/or modify it under
the same terms as the Perl 5 programming language system itself.
 
=cut
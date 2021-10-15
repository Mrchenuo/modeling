classdef schedule_node < handle
   % schedule_node A class to represent a doubly-linked node.
   % Link multiple schedule_node objects together to create linked lists.
   % 时间节点（既包括起点，也包括终点）
   properties
      Data
   end
   properties(SetAccess = private)
      Next = schedule_node.empty
      Prev = schedule_node.empty
      Dest = schedule_node.empty  % 指向航班终点时间节点
   end
   
   methods
      function node = schedule_node(Data, Dest)
         % Construct a schedule_node object
         if nargin > 0
            node.Data = Data;
         end
         
         if ~isempty(Dest)
             node.Dest = Dest;
         end
      end
      
      function insertAfter(newNode, nodeBefore)
         % Insert newNode after nodeBefore.
         removeNode(newNode);
         newNode.Next = nodeBefore.Next;
         newNode.Prev = nodeBefore;
         if ~isempty(nodeBefore.Next)
            nodeBefore.Next.Prev = newNode;
         end
         nodeBefore.Next = newNode;
      end
      
      function insertBefore(newNode, nodeAfter)
         % Insert newNode before nodeAfter.
         removeNode(newNode);
         newNode.Next = nodeAfter;
         newNode.Prev = nodeAfter.Prev;
         if ~isempty(nodeAfter.Prev)
            nodeAfter.Prev.Next = newNode;
         end
         nodeAfter.Prev = newNode;
      end
      
      function removeNode(node)
         % Remove a node from a linked list.
         if ~isscalar(node)
            error('Input must be scalar')
         end
         prevNode = node.Prev;
         nextNode = node.Next;
         if ~isempty(prevNode)
            prevNode.Next = nextNode;
         end
         if ~isempty(nextNode)
            nextNode.Prev = prevNode;
         end
         node.Next = schedule_node.empty;
         node.Prev = schedule_node.empty;
      end
      
      function clearList(node)
         % Clear the list before
         % clearing list variable
         prev = node.Prev;
         next = node.Next;
         removeNode(node)
         while ~isempty(next)
            node = next;
            next = node.Next;
            removeNode(node);
         end
         while ~isempty(prev)
            node = prev;
            prev = node.Prev;
            removeNode(node)
         end
      end
   end
   
   methods (Access = private)
      function delete(node)
         clearList(node)
      end
   end
end
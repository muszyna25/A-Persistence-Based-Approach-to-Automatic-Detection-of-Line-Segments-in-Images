#include <iostream>
#include <cstring>
#include <map>
#include <algorithm>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <iterator>
#include <cinttypes>
#include <numeric>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#define _ITERATOR_DEBUG_LEVEL 0

using namespace boost::filesystem;

using namespace boost;

using namespace std;
using namespace std::chrono;
using namespace cv;

extern "C" {
#include "lsd.h"
}

void Load_Image (string input_file, cv::Point& sizes, cv::Mat_<cv::Vec3b>& input_color, cv::Mat_<cv::Vec3b>& input_mat);
void Write_Image (cv::Mat const& image, string const& name);

const Scalar Red( 0, 0, 255, 1 );
const Scalar White( 255, 255, 255 );
const Scalar Black( 0, 0, 0 );

class Index_Value
{
    
public:
    
    int index;
    
    double value;
    
    Index_Value () {}
    
    Index_Value (int i, double v) { index=i; value=v; }
    
};

bool Decreasing_Values (Index_Value const& iv1, Index_Value const& iv2);

bool Decreasing_Values (Index_Value const& iv1, Index_Value const& iv2) { if ( iv1.value > iv2.value ) return true; else return false; }

struct Compare_Points
{
    bool operator() (cv::Point const& a, cv::Point const& b) const
    {
        if ( a.x < b.x ) return true;
        if ( a.x == b.x ) return ( a.y < b.y );
        return false;
    }
};

class Segment
{
public:
    
    int index, start, finish;
    
    double sum = 0, strength;
    
    std::vector<int> node_indices;
    
    Segment () {}
    
    Segment (int segm_index, Index_Value node, std::vector<int>& segm_indices) // component of a single node
    {
        
        index = segm_index;
        
        sum = node.value;
        
        strength = node.value;
        
        start = node.index;
        
        finish = node.index;
        
        node_indices.assign( 1, node.index );
        
        segm_indices[ node.index ] = segm_index;
    }
    
    void Add_Node (Index_Value node, std::vector<int>& segm_indices)
    {
        sum += node.value;
        
        strength += node.value;
        
        node_indices.push_back( node.index );
        
        start = std::min( start, node.index );
        
        finish = std::max( finish, node.index );
        
        segm_indices[ node.index ] = index;
    }
    
};

int Projection (Point direction, Point p) // along the given direction
{
    if ( direction.x == 0 ) return p.x; // vertical
    
    if ( direction.y == 0 ) return p.y; // horizontal
    
    if ( direction.x == direction.y ) return p.x - p.y; // diagonal down
    
    if ( direction.x + direction.y == 0 ) return p.x + p.y; // diagonal down
    
    if ( direction == Point(-2,1) ) return p.x + 2*p.y;
    if ( direction == Point(2,1) ) return p.x - 2*p.y;
    
    if ( direction == Point(-1,2) ) return p.x + p.y/2;
    if ( direction == Point(1,2) ) return p.x - p.y/2;
    
    std::cout<<"\nError in Projection";
    
    return 0;
}

class Line
{
public:
    double strength;
    int start, finish, projection;
    Point direction, first, second; //endpoints of line
    Line() {}
    Line (Point p0, Point p1, double s) { first = p0; second = p1; strength = s; }
    
    Line (Point initial, Point dir, Segment const& segment)
    {
        projection = Projection(dir, initial);
        direction = dir;
        strength = segment.strength;
        start = segment.start;
        finish = segment.finish;
        first = initial + start * direction;
        second = initial + finish * direction;
    }
    
    void Print ()
    {
        std::cout<<" dir="<<direction<<" s="<<strength<<" "<<first<<second;
    }
    
};

void Merge_Segments (std::map<int,Segment>& segments, int segm_left, int segm_right, Index_Value node, std::vector<int>& segm_indices)
{
    if ( segm_left > segm_right ) swap( segm_left, segm_right );
    
    // Merge the later segment into the earlier segment in the vector segments
    
    segments[ segm_left ].Add_Node( node, segm_indices );
    
    segments[ segm_left ].sum += segments[ segm_right ].sum;
    
    segments[ segm_left ].start = std::min( segments[ segm_left ].start, segments[ segm_right ].start );
    
    segments[ segm_left ].finish = std::max( segments[ segm_left ].finish, segments[ segm_right ].finish );
    
    segments[ segm_left ].node_indices.insert( segments[ segm_left ].node_indices.end(), segments[ segm_right ].node_indices.begin(), segments[ segm_right ].node_indices.end() );
    
    for ( int k = 0; k < segments[ segm_right ].node_indices.size(); k++ )
        segm_indices[ segments[ segm_right ].node_indices[ k ] ] = segm_left;
    
    segments.erase( segm_right );
}


void Copy_Segments (std::vector<Segment>const& segments_old,std::vector<Segment>& segments_new)
{
    segments_new.clear();
    
    for ( auto s : segments_old ) segments_new.push_back( s );
}

bool Find_Persistence (std::map<int,Segment>& segments, double value, double& persistence, std::vector<Segment>& strongest, bool debug)
{
    strongest.clear();
    
    if ( segments.size() == 1 )
    {
        persistence = (segments.begin()->second).strength;
        
        strongest.push_back( segments.begin()->second );
        return true;
    }
    
    //bool debug = false;
    std::vector<Index_Value> strengths;

    persistence = 0;
    
    // Clear vector for each iteration of the external loop.
    strongest.clear();
    
    //if ( segments.size() <= 2 ) return false; // too few segments for comparison
    
    // Copy segment's sum (sum of node values) to variable strength.
    for ( auto it = segments.begin(); it != segments.end(); it++ )
    {
        (it->second).strength = (it->second).sum;   // It assigns sum value to strength value of the same Segment object.
        
        if ( debug ) std::cout<<"seg"<<it->first<<"="<<(it->second).strength<<"i="<<(it->second).index;
        
        // Copy index and strength value of the Segment object.
        strengths.push_back( Index_Value( it->first, (it->second).strength ) );
    }
    
    //Sort them in decreasing order.
    sort( strengths.begin(), strengths.end(), Decreasing_Values );
    
    if ( debug ) for ( auto s : strengths ) std::cout<<" S"<<s.index<<"="<<s.value;
    
    int ind_max = (int)strengths.size()-1;
    double gap_max = strengths.rbegin()->value; //  Initialize the widest gap with the smallest strength value.
    
    for ( int k = 1; k < strengths.size(); k++ )
    {
        if ( gap_max < strengths[k-1].value - strengths[k].value )
        {
            gap_max = strengths[k-1].value - strengths[k].value; // Calculate the size of the gap.
            ind_max = k-1;  //  Save index of the previous strength value.
        }
    }
    
    if ( debug ) std::cout<<" i_max="<<ind_max<<": "<<strengths[ ind_max ].value<<" - "<<strengths[ ind_max + 1 ].value<<" / "<<strengths[ 0 ].value;
    
    // Calculate the persistence.
    persistence = ( strengths[ ind_max ].value - strengths[ ind_max + 1 ].value ); // strengths[ 0 ].value;
    
    for ( int k = 0; k <= ind_max; k++ )
        strongest.push_back( segments[ strengths[k].index ] );
    
    return true;
}

bool Sort (std::vector<double>const& array, std::vector<Index_Value>& sorted, double& median)
{
    sorted.clear();
    
    for ( int i = 0; i < array.size(); i++ )
    {
        if ( array[i] > 0 )
        {
            sorted.push_back( Index_Value( i, array[i] ) );
        }
    }
    
    if ( sorted.size() == 0 ) return false;
    
    sort( sorted.begin(), sorted.end(), Decreasing_Values );
    
    int middle = int( 0.5 * sorted.size() );
    
    median = sorted[ middle ].value;
    
    if ( sorted.size() % 2 == 0 ) median = 0.5 * ( median + sorted[ middle + 1 ].value );
    
    return true;
}

void Find (std::multimap<double,int>& m, double key, int v, std::multimap<double,int>::iterator& it)
{
    auto iters = m.equal_range( key );
    for ( it = iters.first; it != iters.second; ++it )
        if( it->second == v ) return;
    
    cout << "\nError in Find: key="<<key;
    
    return;
}

void Find_Gap (std::multimap<double, std::multimap<double,int>::iterator>& m,
               double key,
               std::multimap<double, int>::iterator& iter,
               std::multimap<double, std::multimap<double,int>::iterator>::iterator& it_return)
{
    auto iters = m.equal_range( key );
    for ( auto it=iters.first; it!=iters.second; ++it )
    {
        if (it->second == iter)
        {
            it_return = it;
            return;
        }
        else
            cout << "Key not found" << endl;
    }
    cout << "\nError in Find_Gap; key=" << key << endl;
    return;
}

void Remove_Gap (std::multimap<double, std::multimap<double,int>::iterator>& gaps_in_strengths, double gap, std::multimap<double,int>::iterator& iter)
{
    std::multimap<double, std::multimap<double,int>::iterator>::iterator it;
    
    Find_Gap(gaps_in_strengths, gap, iter, it);
    
    gaps_in_strengths.erase(it);
}

void Iter_Prev (std::multimap<double, int>& m, multimap<double,int>::iterator& iter, std::multimap<double, int>::iterator& iter_prev)
{
    if ( iter == m.begin() )
        iter_prev = m.end();
    else iter_prev = prev( iter );
}

void Erase_Old_Gaps (std::multimap<double,int>& segments_by_strength,
                        std::multimap<double, std::multimap<double,int>::iterator>& gaps_in_strengths,
                        std::multimap<double, int>::iterator& iter_old )
{
    // Get old strength
    double old_strength = iter_old->first;

    // Get iterator to left element
    std::multimap<double, int>::iterator iter_prev;
    Iter_Prev(segments_by_strength, iter_old, iter_prev);
    
    // Get iterator to right element
    std::multimap<double, int>::iterator iter_next = next( iter_old );
    
    if ( iter_prev == segments_by_strength.end() and iter_next == segments_by_strength.end() )
    {
        Remove_Gap( gaps_in_strengths, old_strength, iter_old );
        return;
    }
    
    double left_gap = old_strength;
    if ( iter_prev != segments_by_strength.end() )
    {
        left_gap = left_gap - (iter_prev->first);
        Remove_Gap( gaps_in_strengths, left_gap, iter_prev );
    }
    
    double right_gap = iter_next->first;
    if ( iter_next != segments_by_strength.end() )
    {
        right_gap = right_gap - old_strength;
        //Remove_Gap( gaps_in_strengths, right_gap, iter_next );
        Remove_Gap( gaps_in_strengths, right_gap, iter_old );
    }
}

void Insert_New_Gaps (std::multimap<double, int>& segments_by_strength,
                      std::multimap<double, multimap<double,int>::iterator>& gaps_in_strengths,
                      multimap<double,int>::iterator& iter_current)
{
    // Strength of current segment
    double new_strength = iter_current->first;
    
    // Get iterator to left element
    std::multimap<double, int>::iterator iter_prev;
    Iter_Prev(segments_by_strength, iter_current, iter_prev);
    
    // Get iterator to right element
    std::multimap<double, int>::iterator iter_next = next( iter_current );
    
    // Insert gap if it is new segment
    if ( iter_prev == segments_by_strength.end() and iter_next == segments_by_strength.end() )
    {
        gaps_in_strengths.insert(std::pair<double, multimap<double,int>::iterator> ( new_strength, iter_current ) );
        return;
    }

    // If the left and right elements exist
    if ( iter_prev != segments_by_strength.end() and iter_next != segments_by_strength.end() )
    {
        Remove_Gap( gaps_in_strengths, iter_next->first - iter_prev->first, iter_prev );
    }
    
    // Add new gap/gaps to gaps_in_strenghts
    
    // Insert left gap
    double left_gap = new_strength - iter_prev->first;
    if ( iter_prev != segments_by_strength.end() )
        gaps_in_strengths.insert(std::pair<double, multimap<double,int>::iterator> ( left_gap, iter_prev ) );
    
    // Insert right gap
    double right_gap = iter_next->first - new_strength;
    if ( iter_next != segments_by_strength.end() )
        gaps_in_strengths.insert(std::pair<double, multimap<double,int>::iterator> ( right_gap, iter_current ) );
}

void Add_New_Segment (std::map<int,Segment>& segments, Index_Value node, int i, std::vector<int>& segm_indices,
                      std::multimap<double, int>& segments_by_strength,
                      std::multimap<double, multimap<double,int>::iterator>& gaps_in_strengths)
{
    // Create new segment with one node
    Segment s( i, node, segm_indices );
    
    // Update strength of new segment with one node.
    //auto iter_new = segments_by_strength.insert( std::pair<double,int>(s.strength, i) );
    std::multimap<double, int>::iterator iter_new = segments_by_strength.insert( std::pair<double,int>(s.strength, i) );
    
    // Insert gap
    Insert_New_Gaps( segments_by_strength, gaps_in_strengths, iter_new );
    
    // Add new segment
    segments.insert( std::make_pair( i, s ) );
}

void Add_New_Node (std::map<int,Segment>& segments, std::vector<int>& segm_indices, int segm_index, Index_Value node,
                   std::multimap<double, int>& segments_by_strength,
                   std::multimap<double, multimap<double,int>::iterator>& gaps_in_strengths)
{
    // Save old strength value of the given segment.
    double old_strength = segments[ segm_index ].strength;
    
    // Add new node to segment
    segments[ segm_index ].Add_Node( node, segm_indices );
    
    std::multimap<double,int>::iterator iter_old;
    
    // Find iterator to segment
    Find( segments_by_strength, old_strength, segm_index, iter_old );
    
    if ( iter_old == segments_by_strength.end() )
        cout << "\nError iterator Add_New_Node()" << endl;

    // Remove gaps
    Erase_Old_Gaps( segments_by_strength, gaps_in_strengths, iter_old );
    
    segments_by_strength.erase( iter_old );
    
    double new_strength = segments[ segm_index ].strength;
    auto iter_new = segments_by_strength.insert( std::pair<double,int>(new_strength, segm_index) );
    
    // Insert new gaps
    Insert_New_Gaps( segments_by_strength, gaps_in_strengths, iter_new );
}

void Merging_Two_Segments (std::map<int,Segment>& segments,
                           std::vector<int>& segm_indices,
                           int segm_right,
                           int segm_left,
                           Index_Value node,
                           std::multimap<double, int>& segments_by_strength,
                           std::multimap<double, multimap<double,int>::iterator>& gaps_in_strengths)
{
    // Save old strength value of the right segment and the left segment.
    double old_strength_segm_left = segments[ segm_left ].strength;
    double old_strength_segm_right = segments[ segm_right ].strength;

    // Erase gaps of left segment
    std::multimap<double,int>::iterator iter_old_segm_left;
    Find( segments_by_strength, old_strength_segm_left, segm_left, iter_old_segm_left );
    
    if ( iter_old_segm_left == segments_by_strength.end() )
        cout << "\nError iterator left segment in Merging_Two_Segments()" << endl;
    
    Erase_Old_Gaps( segments_by_strength, gaps_in_strengths, iter_old_segm_left );
    segments_by_strength.erase( iter_old_segm_left );

    // Erase gaps of right segment
    std::multimap<double,int>::iterator iter_old_segm_right;
    Find( segments_by_strength, old_strength_segm_right, segm_right, iter_old_segm_right );
    
    if ( iter_old_segm_right == segments_by_strength.end() )
        cout << "\nError iterator right segment in Merging_Two_Segments()" << endl;
    
    Erase_Old_Gaps( segments_by_strength, gaps_in_strengths, iter_old_segm_right );
    segments_by_strength.erase( iter_old_segm_right );

    // Merge two segments - the later segment merges into the earlier segment
    Merge_Segments( segments, segm_left, segm_right, node, segm_indices );
    
    int ind_new_segm = segm_left;
    double new_strength = segments[ ind_new_segm ].strength;
    auto iter_new = segments_by_strength.insert( std::pair<double,int>(new_strength, ind_new_segm) );
    
    Insert_New_Gaps( segments_by_strength, gaps_in_strengths, iter_new );
}

bool Find_Best_Segments (std::vector<double>const& graph, std::vector<Segment>& best, double sensitivity, bool debug)
{
    double median;

    std::vector<Index_Value> sorted;
    std::vector<Segment> strongest;
    std::map<int,Segment> segments;
    
    if ( ! Sort( graph, sorted, median ) ) return false; // only zero values
    
    std::vector<int> segm_indices( graph.size(), -1 );
    
    std::multimap<double, int> segments_by_strength;
    std::multimap<double, multimap<double,int>::iterator> gaps_in_strengths; // iter points to the strength just above the gap
    
    int counter = 0;
    
    // Loop over values in the decreasing order
    double gap_max = 0;
    for ( int i = 0; i < sorted.size(); i++ )
    {
        cout << "\nIndex of sorted element: " << i << endl;
        
        int segm_left = -1, segm_right = -1;

        if ( sorted[i].value < sensitivity * median ) break; // too small values
        
        if ( sorted[i].index > 0 ) segm_left = segm_indices[sorted[i].index - 1 ];
        
        if ( sorted[i].index + 1 < segm_indices.size() ) segm_right = segm_indices[ sorted[i].index + 1 ];
        
        if ( segm_left < 0 and segm_right < 0 ) // New isolated node
        {
            Add_New_Segment(segments, sorted[i], counter, segm_indices, segments_by_strength, gaps_in_strengths);
            //Add_New_Segment(segments, sorted[i], i, segm_indices, segments_by_strength, gaps_in_strengths);
            counter++;
            continue;
        }
        
        if ( segm_left >= 0 and segments.find( segm_left ) == segments.end() )
            std::cout<<"\nError in Find_Strongest_Segments: not found l="<<segm_left;
            
        if ( segm_right >= 0 and segments.find( segm_right ) == segments.end() )
            std::cout<<"\nError in Find_Strongest_Segments: not found r="<<segm_right;
                
        if ( segm_left >=0 and segm_right < 0 ) // Add left node to segment
        {
            Add_New_Node (segments, segm_indices, segm_left, sorted[i], segments_by_strength, gaps_in_strengths);
        }
        
        if ( segm_left < 0 and segm_right >= 0 ) // Add right node to segment
        {
            Add_New_Node (segments, segm_indices, segm_right, sorted[i], segments_by_strength, gaps_in_strengths);
        }
        
        if ( segm_left >= 0 and segm_right >= 0 )
        {
            Merging_Two_Segments(segments, segm_indices, segm_right, segm_left, sorted[i], segments_by_strength, gaps_in_strengths);
        }
        
        // Find best segments based on segments_by_strength an gaps_in_strengths
        auto widest = gaps_in_strengths.rbegin();
        if ( gap_max < widest->first )
        {
            gap_max = widest->first;
            best.clear();
            for (std::multimap<double,int>::iterator it = widest->second; it != segments_by_strength.end(); ++it)
                best.push_back( segments[ it->second ] );
        }
    }
    
    return true;
}

bool Find_Strongest_Segments (std::vector<double>const& graph, std::vector<Segment>& best, double sensitivity, bool debug)
{
    //bool debug = false;
    double median;
    double persistence = 0.0, persistence_max = 0;
    
    std::vector<Index_Value> sorted;
    std::vector<Segment> strongest;
    std::map<int,Segment> segments;
    
    if ( ! Sort( graph, sorted, median ) ) return false; // only zero values
    
    double value_max = sorted[0].value;
    
    if ( debug ) std::cout<<" max="<<value_max<<" med="<<median;
    if ( debug ) for ( int k = 0; k < graph.size(); k++ ) std::cout<<"g"<<k<<"="<<graph[k];
    
    std::vector<int> segm_indices( graph.size(), -1 );
    
    // Loop over values in the decreasing order
    for ( int i = 0; i < sorted.size(); i++ )
    {
        int segm_left = -1, segm_right = -1;
        
        if ( sorted[i].value < sensitivity * median) break;
        //if ( sorted[i].value < sensitivity * value_max ) break;
        //if ( sorted[i].value < 0.1 * median ) break; // too small values
        //if ( sorted[i].value < 0.75 * median and sorted[i].value < 0.25 * value_max ) break;
        
        if ( sorted[i].index > 0 ) segm_left = segm_indices[sorted[i].index - 1 ];
        
        if ( sorted[i].index + 1 < segm_indices.size() ) segm_right = segm_indices[ sorted[i].index + 1 ];
        
        if ( debug ) std::cout<<"\ni"<<sorted[i].index<<"v="<<sorted[i].value<<" l="<<segm_left<<" r="<<segm_right;
        
        if ( segm_left < 0 and segm_right < 0 ) // new isolated node
        {
            Segment s( i, sorted[i], segm_indices );
            
            if ( debug ) std::cout<<" new s"<<i;
            segments.insert( std::make_pair( i, s ) );
        }
        
        if ( segm_left >= 0 and segments.find( segm_left ) == segments.end() )
            std::cout<<"\nError in Find_Strongest_Segments: not found l="<<segm_left;
        
        if ( segm_right >= 0 and segments.find( segm_right ) == segments.end() )
            std::cout<<"\nError in Find_Strongest_Segments: not found r="<<segm_right;
        
        if ( segm_left >=0 and segm_right < 0 )
        {
            segments[ segm_left ].Add_Node( sorted[i], segm_indices );
        }
        
        if ( segm_left < 0 and segm_right >= 0 ) segments[ segm_right ].Add_Node( sorted[i], segm_indices );
        
        if ( segm_left >= 0 and segm_right >= 0 ) { Merge_Segments( segments, segm_left, segm_right, sorted[i], segm_indices );
            
        }
        
        if ( debug ) for ( auto s : segments ) std::cout<<"s"<<s.first<<"="<<s.second.sum;
        
        Find_Persistence( segments, sorted[i].value, persistence, strongest, debug );
        
        if ( debug ) std::cout<<" p="<<persistence;
        
        if ( persistence_max < persistence )
        {
            persistence_max = persistence;
            Copy_Segments( strongest, best );
            
            if ( debug ) for ( int k = 0; k < strongest.size(); k++ )
                std::cout<<" s"<<strongest[k].index;
        }
    }
    
    if ( best.size() == 0 )
        Copy_Segments( strongest, best ); // only one segment found until we hit the median value
    
    return true;
    
}

bool Find_Strongest_Line (std::map< Point, std::map< int, Line >,
                           Compare_Points >& map_lines, int offset, std::map< Point, Line,
                           Compare_Points >& strongest_lines, Point& direction_best, Line& line_best)
{
    
    bool debug = false;
    double strength_max = 0, strength = 0;
    
    // Find the current best line segment
    for ( auto iter = map_lines.begin(); iter != map_lines.end(); iter++ ) // loop over all directions
    {
        Point direction = iter->first;
        
        auto it = strongest_lines.find( direction );
        
        if ( debug ) std::cout<<"\n\nd="<<direction;
        
        if ( it != strongest_lines.end() ) strength = (it->second).strength;
        else
        {
            strength = 0;
            
            for ( auto it_line = (iter->second).begin(); it_line != (iter->second).end(); it_line++ ) // loop over all lines in the direction
                if ( strength < (it_line->second).strength )
                {
                    strength = (it_line->second).strength;
                    
                    strongest_lines[ direction ] = it_line->second;
                    
                    if ( debug ) { std::cout<<"\n\nstr="<<strength<<"cd="<<direction; for ( auto l : strongest_lines ) l.second.Print(); }
                        
                }
        }
            
        if ( strength_max < strength )
        {
            strength_max = strength; direction_best = direction;

            if ( debug ) { std::cout<<" sm="<<strength_max<<"db="<<direction_best; for ( auto l : strongest_lines ) l.second.Print(); }
                
        }
            
    }
        
    if ( strength_max == 0 ) return false;
    
    line_best = strongest_lines[ direction_best ];
    
    if ( line_best.direction == Point(0,0) )
    {
        std::cout<<"\nError in Find_Strongest_Line in the middle:dir="<<direction_best<<" s="<<strength_max<<"\n";
        
        for ( auto l : strongest_lines ) l.second.Print();
    }

    // Remove other line segments that are too close to the best line segment (including the segment itself)
    
    for ( int p = line_best.projection - offset + 1; p < line_best.projection + offset; p++ )
    {
        auto it  = map_lines[ direction_best ].find( p );
        
        if ( it == map_lines[ direction_best ].end() ) continue; // no close line
        
        if ( (it->second).finish < line_best.start ) continue; // this line finishes before the best line starts
        
        if ( (it->second).start > line_best.finish ) continue; // this line starts after the best line finishes
        
        map_lines[ direction_best ].erase( it ); // this line is too close to the best line, also removing the best line
    }
    
    if ( debug ) { std::cout<<"\nbest: "; line_best.Print(); std::cout<<" pr="<<line_best.projection; }

    // Erase the line segments that intersect the current best line segment
    
    if ( debug ) { std::cout<<"\nBefore erase:"; for ( auto l : strongest_lines ) l.second.Print(); }
    
    for ( auto iter = map_lines.begin(); iter != map_lines.end(); iter++ ) // loop over all directions
    {
        Point direction = iter->first;
        
        if ( direction == direction_best ) continue; // the best direction was already checked above
        
        if ( debug ) std::cout<<"\n direction="<<direction; //<<"dir_ort="<<dir_ort;
        
        int start = Projection( direction, line_best.first );
        
        int finish = Projection( direction, line_best.second );
        
        if ( start > finish ) swap( start, finish );
        
        if ( debug ) std::cout<<" st="<<start<<" f="<<finish;
        
        for ( int p = start + 1; p < finish; p++ )
        {
            auto it  = map_lines[ direction ].find( p );
            
            if ( it == map_lines[ direction ].end() ) continue; //this line can't meet the best line
            
            if ( debug ) { std::cout<<"\n  try p="<<p; (it->second).Print(); }
            
            int p1 = Projection( direction_best, (it->second).first );
            
            int p2 = Projection( direction_best, (it->second).second );
            
            if ( debug ) std::cout<<" p1="<<p1<<" p2="<<p2;
            
            if ( line_best.projection <= p1 and line_best.projection <= p2 ) continue;
            
            if ( line_best.projection >= p1 and line_best.projection >= p2 ) continue;
            
            auto it_line = strongest_lines.find( direction );
            
            if ( it_line != strongest_lines.end() and (it_line->second).projection == p )
            {
                strongest_lines.erase( it_line );
                if ( debug ) std::cout<<" erased from strongest_lines";
            }
            
            map_lines[ direction ].erase( it );
            
            if ( debug ) std::cout<<" erased from map_lines";
        }
    }
    
    if ( debug ) { std::cout<<"\nBefore final erase:"; for ( auto l : strongest_lines ) l.second.Print(); }
    
    strongest_lines.erase( direction_best ); // remove the best line from any later search
    
    if ( line_best.direction == Point(0,0) ) std::cout<<"\nError in Find_Strongest_Line at the end";
        
        if ( debug ) for ( auto l : strongest_lines ) l.second.Print();
    
    return true;
}

bool Point_Near_Boundary (Point sizes, Point point, int length)
{
    if ( point.x < length or point.y < length ) return true;
    
    if ( point.x + length >= sizes.x ) return true;
    
    if ( point.y + length >= sizes.y ) return true;
    
    return false;
}

// vector of matrix values from a pixel corner along a direction
void Find_Differences (Mat_<Vec3b>const& matrix, Point point, Point line, int size, std::vector<Vec3d>& differences)
{
    differences.clear();
    differences.assign( size, 0 );
    if ( Point_Near_Boundary( matrix.size(), point, size ) ) return;
    for ( int i = 0; i < size; i++ )
    {
        if ( line.y == 0 ) // differences across a horizontal line
        {
            differences[ i ] = 0.5 * ( (Vec3d)matrix( point.y + i, point.x ) + (Vec3d)matrix( point.y + i, point.x - 1 )
                                      - (Vec3d)matrix( point.y - i - 1, point.x ) - (Vec3d)matrix( point.y - i - 1, point.x - 1 ) );
        }
        if ( line.x == 0 ) // differences across a vertical line
        {
            differences[ i ] = 0.5 * ( (Vec3d)matrix( point.y, point.x + i ) + (Vec3d)matrix( point.y - 1, point.x + i )
                                      - (Vec3d)matrix( point.y, point.x - i - 1 ) - (Vec3d)matrix( point.y - 1, point.x - i - 1 ) );
        }
        if ( line.x == line.y ) // differences across a diagonal down
        {
            differences[ i ] =  (Vec3d)matrix( point.y - i - 1, point.x + i ) - (Vec3d)matrix( point.y + i, point.x - i - 1 )
            + 0.5 * ( (Vec3d)matrix( point.y - i - 2, point.x + i ) + (Vec3d)matrix( point.y - i - 1, point.x + i + 1 ) )
            - 0.5 * ( (Vec3d)matrix( point.y + i, point.x - i - 2 ) + (Vec3d)matrix( point.y + i + 1, point.x - i - 1 ) );
        }
        if ( line.x + line.y == 0 )  // differences across a diagonal up
        {
            differences[ i ] =  (Vec3d)matrix( point.y + i, point.x + i ) - (Vec3d)matrix( point.y - i - 1, point.x - i - 1 )
            + 0.5 * ( (Vec3d)matrix( point.y + i + 1, point.x + i ) + (Vec3d)matrix( point.y + i, point.x + i + 1 ) )
            - 0.5 * ( (Vec3d)matrix( point.y - i - 2, point.x - i - 1) + (Vec3d)matrix( point.y - i - 1, point.x - i - 2 ) );
        }
    }
}

int Color_Norm (Vec3i v){ return abs( v[0] ) + abs( v[1] ) + abs( v[2] ); }

int Color_Norm_Inf (Vec3i v) { return max ( abs(v[0]), max( abs(v[1]), abs(v[2])) ); }

void Find_Gradient (Mat_<Vec3b>const& matrix, Point point, Point line, std::vector<double>const& coefficients, double& gradient)
{
    Vec3d difference = Vec3d(0,0,0);
    std::vector<Vec3d> differences;
    Find_Differences( matrix, point, line, (int)coefficients.size(), differences );
    for ( int k = 0; k < differences.size(); k++ ) difference += differences[ k ] * coefficients[k];
    gradient = Color_Norm_Inf( difference );
    //gradient = Color_Norm( difference );
}

void Find_Gradient (Mat_<Vec3b>const& matrix, Point initial, Point direction, std::vector<double>const& coefficients, std::vector<double>& gradient)
{
    bool debug = false; //true;
    int size = (int)coefficients.size();
    if ( direction.y == 0 ) // horizontal
    {
        if ( debug ) std::cout<<"\ny="<<initial.y;
        gradient.assign( matrix.cols + 1, 0 );
        for ( int k = size + 1; k + size + 1 < gradient.size(); k++ )
            Find_Gradient( matrix, Point( k, initial.y ), direction, coefficients, gradient[ k ] );
    }
    if ( direction.x == 0 ) // vertical
    {
        if ( debug ) std::cout<<"\nx="<<initial.x;
        gradient.assign( matrix.rows + 1, 0 );
        for ( int i = size + 1; i + size + 1 < gradient.size(); i++ )
            Find_Gradient( matrix, Point( initial.x, i ), direction, coefficients, gradient[ i ] );
    }
    if ( direction.x == direction.y ) // diagonal down
    {
        if ( debug ) std::cout<<"\nx-y="<<initial.x-initial.y;
        gradient.assign( std::min( matrix.rows, matrix.cols ) + 1, 0 );
        for ( int k = size + 1; k + size + 1 < gradient.size(); k++ )
            Find_Gradient( matrix, Point( initial.x + k, initial.y + k ), direction, coefficients, gradient[ k ] );
    }
    if ( direction.x + direction.y == 0 ) // diagonal up
    {
        gradient.assign( std::min( matrix.rows, matrix.cols ) + 1, 0 );
        for ( int k = size + 1; k + size + 1 < gradient.size(); k++ )
            Find_Gradient( matrix, Point( initial.x - k, initial.y + k ), direction, coefficients, gradient[ k ] );
    }
}

// vector of matrix values from a pixel corner along a direction
void Find_Differences (Mat_<Vec3b>const& matrix, Point point, Point direction, std::vector<Vec3d>& differences)
{
    if ( Point_Near_Boundary( matrix.size(), point, (int)differences.size() ) ) return;
    
    for ( int i = 0; i < differences.size(); i++ )
    {
        if ( direction.y == 0 ) // differences across a horizontal line
        {
            differences[ i ] = 1.5 * ( (Vec3d)matrix( point.y + i, point.x ) + (Vec3d)matrix( point.y + i, point.x - 1 )
                                      - (Vec3d)matrix( point.y - i - 1, point.x ) - (Vec3d)matrix( point.y - i - 1, point.x - 1 ) );
        }
        else if ( direction.x == 0 ) // differences across a vertical line
        {
            differences[ i ] = 1.5 * ( (Vec3d)matrix( point.y, point.x + i ) + (Vec3d)matrix( point.y - 1, point.x + i )
                                      - (Vec3d)matrix( point.y, point.x - i - 1 ) - (Vec3d)matrix( point.y - 1, point.x - i - 1 ) );
        }
        else if ( direction.x == direction.y ) // differences across a diagonal down
        {
            differences[ i ] =  1.5 * ( (Vec3d)matrix( point.y - i - 1, point.x + i ) - (Vec3d)matrix( point.y + i, point.x - i - 1 ) )
                                        + 0.75 * ( (Vec3d)matrix( point.y - i - 2, point.x + i ) + (Vec3d)matrix( point.y - i - 1, point.x + i + 1 ) )
                                        - 0.75 * ( (Vec3d)matrix( point.y + i, point.x - i - 2 ) + (Vec3d)matrix( point.y + i + 1, point.x - i - 1 ) );
        }
        else if ( direction.x + direction.y == 0 )  // differences across a diagonal up
        {
            differences[ i ] = 1.5 * ( (Vec3d)matrix( point.y + i, point.x + i ) - (Vec3d)matrix( point.y - i - 1, point.x - i - 1 ) )
                                        + 0.75 * ( (Vec3d)matrix( point.y + i + 1, point.x + i ) + (Vec3d)matrix( point.y + i, point.x + i + 1 ) )
                                        - 0.75 * ( (Vec3d)matrix( point.y - i - 2, point.x - i - 1) + (Vec3d)matrix( point.y - i - 1, point.x - i - 2 ) );
        }
        else if ( direction == Point(-2,1) )
        {
            differences[ i ] = (2-i%2) * ( (Vec3d)matrix( point.y + i, point.x + (i+1)/2 ) - (Vec3d)matrix( point.y - i - 1, point.x - (i+1)/2 - 1 ) ) + (1+i%2) * ( (Vec3d)matrix( point.y + i, point.x + (i+1)/2-1 ) - (Vec3d)matrix( point.y - i - 1, point.x - (i+1)/2 ) );
        }
        else if ( direction == Point(2,1) )
        {
            differences[ i ] = (2-i%2) * ( (Vec3d)matrix( point.y - i-1, point.x + (i+1)/2 ) - (Vec3d)matrix( point.y + i, point.x - (i+1)/2 - 1 ) ) + (1+i%2) * ( (Vec3d)matrix( point.y - i-1, point.x + (i+1)/2-1 ) - (Vec3d)matrix( point.y + i, point.x - (i+1)/2 ) );
        }
        else if ( direction == Point(1,2) )
        {
            differences[ i ] = (2-i%2) * ( (Vec3d)matrix( point.y + (i+1)/2, point.x -i-1 ) - (Vec3d)matrix( point.y - (i+1)/2 -1, point.x +i ) ) + (1+i%2) * ( (Vec3d)matrix( point.y + (i+1)/2-1, point.x -i-1) - (Vec3d)matrix( point.y - (i+1)/2, point.x +i ) );
        }
        else if ( direction == Point(-1,2) )
        {
            differences[ i ] = (2-i%2) * ( (Vec3d)matrix( point.y + (i+1)/2, point.x +i ) - (Vec3d)matrix( point.y - (i+1)/2 -1, point.x -i-1 ) ) + (1+i%2) * ( (Vec3d)matrix( point.y + (i+1)/2-1, point.x +i) - (Vec3d)matrix( point.y - (i+1)/2, point.x -i-1 ) );
        }
    }
    
}

//-----------------------------
bool Derivative_at_Point (Mat_<Vec3b>const& image, Point point, Point direction, std::vector<double>const& weights, double& derivative, bool debug)
{
    std::vector<Vec3d> differences( weights.size(), 0 );
    
    Find_Differences( image, point, direction, differences );
    
    Vec3d difference = Vec3d(0,0,0);
    
    for ( int k = 0; k < differences.size(); k++ )
        
        difference += weights[k] * differences[ k ];
    
    derivative = Color_Norm( difference );
    
    return true;
}

bool Derivatives_along_Line (Mat_<Vec3b>const& image, Point initial, Point direction, int offset, int length_min, std::vector<double>const& weights, std::vector<double>& derivatives, bool debug)
{
    int s = 0;
    
    int b = length_min + offset;
    
    if ( direction.y == 0 ) // horizontal
    {
        if ( initial.y < offset or image.rows < offset + initial.y )
            return false; // too close to the boundary
        
        s = image.cols + 1;
    }
    else if ( direction.x == 0 ) // vertical
    {
        if ( initial.x < offset or image.cols < offset + initial.x )
            return false; // too close to the boundary
        
        s = image.rows + 1;
    }
    else if ( direction.x == direction.y ) // diagonal down
    {
        if ( initial.y < b - image.rows or image.cols - b < initial.x ) return false;
        
        s = std::min( image.rows, image.cols ) + 1;
        
        if ( s > image.rows - initial.y ) s = image.rows - initial.y;
        
        if ( s > image.cols - initial.x ) s = image.cols - initial.x;
    }
    else if ( direction.x + direction.y == 0 ) // diagonal up
    {
        if ( initial.x < b or image.rows - b < initial.y ) { return false; }
        
        s = std::min( image.rows, image.cols ) + 1;
        
        if ( initial.y == 0 and s > initial.x ) s = initial.x;
        
        if ( initial.x == image.cols and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(-2,1) )
    {
        s = std::min( image.rows+1, (image.cols+1)/2 );
        
        if ( initial.x < 2*b or image.rows - b < initial.y ) { return false; }
        
        if ( initial.y == 0 and s > 2*initial.x ) s = 2*initial.x;
        
        if ( initial.x == image.cols and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(2,1) )
    {
        s = std::min( image.rows+1, (image.cols+1)/2 );
        
        if ( initial.x > image.cols - 2*b or initial.y < b - image.rows ) { return false; }
        
        if ( initial.y == 0 and 2*s > image.cols - initial.x ) s = (image.cols - initial.x) / 2;
        
        if ( initial.x == 0 and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(1,2) )
    {
        s = std::min( (image.rows+1)/2, image.cols+1 );
        
        if ( initial.x > image.cols - b or initial.y < 2*b - image.rows ) { return false; }
        
        if ( initial.y == 0 and s > image.cols - initial.x ) s = image.cols - initial.x;
        
        if ( initial.x == 0 and 2*s > image.rows - initial.y ) s = (image.rows - initial.y) / 2;
    }
    else if ( direction == Point(-1,2) )
    {
        s = std::min( (image.rows+1)/2, image.cols+1 );
        
        if ( initial.x < b or image.rows - 2*b < initial.y ) { return false; }
        
        if ( initial.y == 0 and s > initial.x ) s = initial.x;
        
        if ( initial.x == image.cols and 2*s > image.rows - initial.y ) s = (image.rows - initial.y) / 2;
    }
    
    if ( s <= length_min + 2 * offset - 1 ) return false;
    
    derivatives.assign( s - 2 * offset + 1, 0 );
    
    for ( int k = offset; k <= s - offset; k++ )
        Derivative_at_Point( image, initial + k * direction, direction, weights, derivatives[ k - offset ], debug );
    
    return true;
}

//-----------------------------
bool Find_Persistent_Segments (Mat_<Vec3b>const& matrix, Point initial, Point direction, int offset, int length_min, std::vector<double>const& weights, std::map< int, Line >& lines, double sensitivity)
{
    bool debug = false; //true;
    
    std::vector<double> derivatives;
    
    if ( direction.y != 0 ) offset /= abs( direction.y );
    
    Derivatives_along_Line( matrix, initial, direction, offset, length_min, weights, derivatives, debug );
    
    std::vector<Segment> segments;
    
    if ( ! Find_Strongest_Segments( derivatives, segments, sensitivity, debug ) )
        return false;
    
    // Filter segments by length
    for ( int k = 0; k < segments.size(); k++ )
    {
        if ( segments[k].finish - segments[k].start < length_min ) continue;
        
        segments[k].start += offset;
        
        segments[k].finish += offset;
        
        Line line( initial, direction, segments[k] );
        
        lines.insert( std::make_pair( line.projection, line ) );
    }
    
    if ( lines.size() == 0 ) return false; // no strong lines found
    
    return true;
}

bool Derivatives_Initialization (Mat_<Vec3d>const& image, Point initial, Point direction, int offset, int length_min, std::vector<double>& derivatives, bool debug)
{
    int s = 0;
    
    int b = length_min + offset;
    
    if ( direction.y == 0 ) // horizontal
    {
        if ( initial.y < offset or image.rows < offset + initial.y )
            return false; // too close to the boundary
        
        s = image.cols + 1;
    }
    else if ( direction.x == 0 ) // vertical
    {
        if ( initial.x < offset or image.cols < offset + initial.x )
            return false; // too close to the boundary
        s = image.rows + 1;
    }
    else if ( direction.x == direction.y ) // diagonal down
    {
        if ( initial.y < b - image.rows or image.cols - b < initial.x ) return false;
        
        s = std::min( image.rows, image.cols ) + 1;
        
        if ( s > image.rows - initial.y ) s = image.rows - initial.y;
        
        if ( s > image.cols - initial.x ) s = image.cols - initial.x;
    }
    else if ( direction.x + direction.y == 0 ) // diagonal up
    {
        if ( initial.x < b or image.rows - b < initial.y ) { return false; }
        
        s = std::min( image.rows, image.cols ) + 1;
        
        if ( initial.y == 0 and s > initial.x ) s = initial.x;
        
        if ( initial.x == image.cols and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(-2,1) )
    {
        s = std::min( image.rows+1, (image.cols+1)/2 );
        
        if ( initial.x < 2*b or image.rows - b < initial.y ) { return false; }
        
        if ( initial.y == 0 and s > 2*initial.x ) s = 2*initial.x;
        
        if ( initial.x == image.cols and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(2,1) )
    {
        s = std::min( image.rows+1, (image.cols+1)/2 );
        
        if ( initial.x > image.cols - 2*b or initial.y < b - image.rows ) { return false; }
        
        if ( initial.y == 0 and 2*s > image.cols - initial.x ) s = (image.cols - initial.x) / 2;
        
        if ( initial.x == 0 and s > image.rows - initial.y ) s = image.rows - initial.y;
    }
    else if ( direction == Point(1,2) )
    {
        s = std::min( (image.rows+1)/2, image.cols+1 );
        
        if ( initial.x > image.cols - b or initial.y < 2*b - image.rows ) { return false; }
        
        if ( initial.y == 0 and s > image.cols - initial.x ) s = image.cols - initial.x;
        
        if ( initial.x == 0 and 2*s > image.rows - initial.y ) s = (image.rows - initial.y) / 2;
    }
    
    else if ( direction == Point(-1,2) )
    {
        s = std::min( (image.rows+1)/2, image.cols+1 );
        
        if ( initial.x < b or image.rows - 2*b < initial.y ) { return false; }
        
        if ( initial.y == 0 and s > initial.x ) s = initial.x;
        
        if ( initial.x == image.cols and 2*s > image.rows - initial.y) s = (image.rows - initial.y) / 2;
    }
    
    if ( s <= length_min + 2 * offset - 1 ) return false;
    
    if ( debug ) std::cout<<"\nsize="<<s;
    
    derivatives.assign( s - 2 * offset + 1, 0 );
    
    return true;
}


bool Derivatives (Mat_<Vec3d>const& gradient_x, Mat_<Vec3d>const& gradient_y, Point initial, Point direction, int offset, int length_min, std::vector<double>& derivatives)
{
    bool debug = false;
    
    double l = norm( direction );
    
    if ( ! Derivatives_Initialization( gradient_x, initial, direction, offset, length_min, derivatives, false ) ) return false;
    
    Point p;
    
    if(direction == Point(0,1) and initial.x <= 91 and 89 <= initial.x ) debug = true; else debug = false;
    if(debug) cout << "\ndir" << direction << " intial point " << initial;
    
    for ( int k = 0; k < derivatives.size(); k++ )
    {
        p = initial + (k+offset) * direction;
        
        derivatives[ k ] = Color_Norm_Inf( -direction.y * gradient_x.at<Vec3d>( p ) + direction.x * gradient_y.at<Vec3d>( p ) ) / l;
        
        if(debug) cout << "\nk=" << k << " " << "gx " << gradient_x.at<Vec3d>( p ) << "gy " << gradient_y.at<Vec3d>( p ) << " " << derivatives[ k ];
    }
    
    return true;
}


bool Persistent_Segments (Mat_<Vec3d>const& gradient_x, Mat_<Vec3d>const& gradient_y, Point initial, Point direction, int offset, int length_min, std::map< int, Line >& lines, double contrast_ratio)
{
    bool debug = false; //true;
    
    std::vector<double> derivatives;
    
    if ( ! Derivatives( gradient_x, gradient_y, initial, direction, offset, length_min, derivatives ) ) return false;
    
    if(direction == Point(0,1) and initial.x == 91 ) debug = true; else debug = false;
    
    std::vector<Segment> segments;
    
    if ( ! Find_Strongest_Segments( derivatives, segments, contrast_ratio, debug ) )
    //if ( ! Find_Best_Segments( derivatives, segments, contrast_ratio, debug ) )
        return false;
    
    // Filter segments by length
    for ( int k = 0; k < segments.size(); k++ )
    {
        if ( segments[k].finish - segments[k].start < length_min ) continue;
        
        segments[k].start += offset;
        
        segments[k].finish += offset;
        
        Line line( initial, direction, segments[k] );
        
        if ( debug ) std::cout<<"\ns="<<line.strength<<" p="<<line.projection<<" ["<<line.start<<","<<line.finish<<"]";
        
        lines.insert( std::make_pair( line.projection, line ) );
    }
    
    if ( lines.size() == 0 ) return false; // no strong lines found
    
    return true;
}


//=========================
bool Image_Gradient_x (Mat_<Vec3b>const& image, int offset, Mat_<Vec3d>& gradient_x)
{
    cout << "Offset " << offset << endl;
    
    for ( int i = offset; i < image.rows - offset; i++ )
        for ( int j = offset; j < image.cols - offset; j++ )
        {
            gradient_x( i, j ) = (Vec3d)image.at<Vec3b>( i-1, j ) + (Vec3d)image.at<Vec3b>( i, j ) - (Vec3d)image.at<Vec3b>( i-1, j-1 ) - (Vec3d)image.at<Vec3b>( i, j-1 );
            
            if( j == 91)
            {
                cout << "\ni=" << i << " grdx=" << gradient_x( i, j ) << "="<<(Vec3d)image.at<Vec3b>( i-1, j )
                        <<"+"<<(Vec3d)image.at<Vec3b>( i, j )
                        <<"-"<< (Vec3d)image.at<Vec3b>( i-1, j-1 )
                <<"-"<<(Vec3d)image.at<Vec3b>( i, j-1 );
            }
        }
    
    return true;
}


//=========================
bool Image_Gradient_y (Mat_<Vec3b>const& image, int offset, Mat_<Vec3d>& gradient_y)
{
    for ( int i = offset; i < image.rows - offset; i++ )
        for ( int j = offset; j < image.cols - offset; j++ )
            gradient_y( i, j ) = (Vec3d)image.at<Vec3b>( i, j-1 ) + (Vec3d)image.at<Vec3b>( i, j ) - (Vec3d)image.at<Vec3b>( i-1, j-1 ) - (Vec3d)image.at<Vec3b>( i-1, j );
    return true;
}

void Find_Lines (cv::Mat_<cv::Vec3b>const& image, std::vector<Line>& lines, int num_lines, int offset, Mat& fimage, double sensitivity)
{
    Point ksize( 3, 3 ); // coordinates should be odd
    //double sigma = 0; // then the default is 0.3*((ksize-1)*0.5 - 1) + 0.8
    //GaussianBlur( input_color, input_color, ksize, sigma, sigma );
    
    Mat_<Vec3d> gradient_x( image.size(), Vec3d(0,0,0) ), gradient_y( image.size(), Vec3d(0,0,0) );
    
    Image_Gradient_x( image, offset, gradient_x ); // functions below
    Image_Gradient_y( image, offset, gradient_y );
    
    // Parameters
    int length_min = offset;
    // old parameters
    int length = 16; //max( 8, int( 1 * side ) ); // for the rectangular mask at a strong edge
    int width = 4; //min( 4, int( 0.5 * length ) );
    Point box_x( length, width );
    Point box_y( width, length );
    //double sensitivity = 1.5; // ratio of value_max for threshold
    Point initial, direction;
    
    std::vector<Point> directions{ Point(1,0), Point(0,1), Point(1,1), Point(-1,1), Point(-2,1), Point(2,1), Point(1,2), Point(-1,2) }; //
    std::map< Point, std::vector<Point>, Compare_Points> initial_points;
    
    for ( auto dir : directions )
    {
        if ( dir.y == 0 ) // horizontal
            for ( int y = offset + 1; y + offset + 1 < image.rows; y += 1 )
                initial_points[ dir ].push_back( Point( 0, y ) );
        else if ( dir.x == 0 ) // vertical
            for ( int x = offset + 1; x + offset + 1 < image.cols; x += 1 )
                initial_points[ dir ].push_back( Point( x, 0 ) );
        else if ( dir.x == dir.y or dir == Point(2,1) or dir == Point(1,2) ) // diagonal down
            for ( int x = offset + 1 - image.rows * abs( dir.x ); x + offset + 1 < image.cols; x += 1 )
            {
                if ( x % abs( dir.x ) != 0 ) continue;
                if ( x >= 0 ) initial_points[ dir ].push_back( Point(x, 0 ) ); // top horizontal side
                else initial_points[ dir ].push_back( Point( 0, -x / abs( dir.x ) ) ); // left vertical side
            }
        else if ( dir.x + dir.y == 0 or dir == Point(-2,1) or dir == Point(-1,2) ) // diagonal up
            for ( int x = offset + 1; x + offset + 1 < image.cols + image.rows * abs( dir.x ); x += 1 )
            {
                if ( x % abs( dir.x ) != 0 ) continue;
                
                if ( x < image.cols ) initial_points[ dir ].push_back( Point( x, 0 ) ); // top horizontal side
                else initial_points[ dir ].push_back( Point(image.cols, (x - image.cols) / abs( dir.x ) ) ); // right vertical side
            }
    }
    
    std::map< Point, std::map< int, Line >, Compare_Points > map_lines; // 1st = direction, 1st in 2nd = 1D projection along the line
    std::vector<double> weights{ 1, 0.5 };
    
    for ( auto direction : directions )
        for ( auto initial : initial_points[ direction ] )
            //Find_Persistent_Segments( image, initial, direction, offset, length_min, weights, map_lines[ direction ], sensitivity );
            Persistent_Segments( gradient_x, gradient_y, initial, direction, offset, length_min, map_lines[ direction ], sensitivity );
            
    // Draw lines
    lines.clear();
    Line line_best;
    Point direction_best;
    std::map< Point, Line, Compare_Points > strongest_lines;
    
    for ( int k = 0; k < num_lines; k++ )
        if ( Find_Strongest_Line( map_lines, offset, strongest_lines, direction_best, line_best ) )
        {
            cout << "\n" << k;
            line_best.Print();
            lines.push_back( line_best );
        }
    
    Mat img = image.clone();
    fimage = img;
}

void Draw_Lines (Mat& img ,vector<Line>& lines)
{
    for ( int k = 0; k < lines.size(); k++ )
    {
        line( img, lines[ k ].first, lines[ k ].second, Red, 2, CV_AA );
        circle( img, lines[ k ].first, 3.0, Scalar( 255, 0, 0 ), -1, 8 );
        circle( img, lines[ k ].second, 3.0, Scalar( 255, 0, 0 ), -1, 8 );
    }
    
    /*
    for ( int k = 0; k < lines.size(); k++ )
    {
        line( img, 4*lines[ k ].first, 4*lines[ k ].second, Red, 1, CV_AA );
        circle( img, 4*lines[ k ].first, 2.0, Scalar( 255, 0, 0 ), -1, 8 );
        circle( img, 4*lines[ k ].second, 2.0, Scalar( 255, 0, 0 ), -1, 8 );
    }
    */
}

/* Save image to file */
void Write_Image (cv::Mat const& image, string const& name)
{
    try
    {
        imwrite( name, image );
    }
    catch (std::runtime_error& ex) { fprintf(stderr, "Exception converting to PNG: %s\n", ex.what()); }
}

/* Load image from file */
void Load_Image (string input_file, cv::Point& sizes, cv::Mat_<cv::Vec3b>& input_color, cv::Mat_<cv::Vec3b>& input_mat)
{
    input_color = cv::imread( input_file, CV_LOAD_IMAGE_COLOR );
    if ( input_color.empty() ) { cout<<"Image not found: ["<<input_file<<"]\n"; exit(0); }
    input_mat = input_color.clone();
    sizes.x = input_mat.cols;
    sizes.y = input_mat.rows;
}

inline bool exist(const std::string& name)
{
    std::ifstream file(name);
    if(!file)            // If the file was not found, then file is 0, i.e. !file=1 or true.
        return false;    // The file was not found.
    else                 // If the file was found, then file is non-0.
        return true;     // The file was found.
}

bool Read_BSD_Human (std::string path, vector<Point2d>& boundary)
{
    std::ifstream file;
    
    if( ! exist(path) )
    {
        std::cout<<"\nFile "<<path<<" not found\n";
        return false;
    }

    // Open file
    file.open( path );

    // Array with dimensions
    int32_t* dims = new int32_t[2];

    file.read((char*)dims, sizeof(int32_t)*2);

    int32_t width = dims[0];

    int32_t height = dims[1];

    boundary.clear();

    // Find number of pixels
    int num_pixels = width * height;

    int32_t* data = new int32_t[ num_pixels ];

    uchar* bdata = new uchar[ num_pixels ];

    file.read( (char*)data, num_pixels * sizeof(int32_t) ) ;

    file.read( (char*)bdata, num_pixels * sizeof(uchar) );
    
    Mat_<uchar> boundary_mesh = Mat_<uchar>( height, width, bdata ).clone();

    for ( int i = 0; i < height; i++ )
    {
        for ( int j = 0; j < width; j++ )
        {
            if ( boundary_mesh( i, j ) == 1 )
                boundary.push_back( Point2d( j+0.5, i+0.5 ) );
        }
    }

    delete data;

    delete bdata;
    
    delete dims;
    
    file.close();
    
    return true;
}

int Find_File_Names (string file_name, string path)
{
    vector<int> nb_of_human_segmentations;
    
    string regex_pattern = "^" + file_name + "_\\d+.dat$";
    
    boost::regex reg(regex_pattern);
    
    for(recursive_directory_iterator it( path ); it != recursive_directory_iterator(); ++it)
    {
        std::string name = it->path().filename().string();
        
        if(boost::regex_search(name, reg))
        {
            string matched = it->path().filename().string();
            std::size_t pos = matched.find("_");
            std::string str = matched.substr (pos+1,1);
            
            nb_of_human_segmentations.push_back(stoi(str));
        }
    }
    
    int max = *max_element(nb_of_human_segmentations.begin(), nb_of_human_segmentations.end());
    
    cout << "Number of human segmentations: " << max << endl;
    
    return max;
}

void Compute_Coefficients(vector<int32_t>& coefficients, Point p1, Point p2)
{
    int32_t a, b, c;
    
    a = (int32_t)p1.y - (int32_t)p2.y;
    coefficients.push_back(a);
    
    b = (int32_t)p2.x - (int32_t)p1.x;
    coefficients.push_back(b);
    
    c = ((int32_t)p1.x - (int32_t)p2.x) * (int32_t)p1.y + ((int32_t)p2.y - (int32_t)p1.y) * (int32_t)p1.x;
    
    coefficients.push_back(c);
}

void Find_Line_Equation (vector<Line>& lines, map<int, vector<int32_t>>& equations_parameters)
{
    int num_line = 0;
    
    for (std::vector<Line>::iterator l_it = lines.begin(), l_end = lines.end(); l_it != l_end; ++l_it)
    {
        Point line_p1 = (*l_it).first;
        Point line_p2 = (*l_it).second;
        
        // Find coefficients for general form of line equation
        // double a, b, c in a vector
        vector<int32_t> coefficients;
        Compute_Coefficients(coefficients, line_p1, line_p2);
        
        equations_parameters.insert(make_pair(num_line, coefficients));
        
        num_line++;
    }
}

double Compute_Distance_To_Line (Point2d point, vector<int32_t>& parameters)
{
    double dist = 0;
    
    int32_t a = parameters[0];
    int32_t b = parameters[1];
    int32_t c = parameters[2];
    
    dist = ( abs(a * point.x + b * point.y + c) ) / sqrt( pow(a, 2) + powf(b, 2) );
    
    return dist;
}

// Check if a boundary point is outside a box with offset eps = 2
bool Check_If_Outside_Boundary_Box (Point2d point, Point2d first, Point2d second, double eps)
{
    Point2d btm_left_corner = Point2d( min(first.x, second.x) - eps, min(first.y, second.y) - eps );
    Point2d upp_right_corner = Point2d( max(first.x, second.x) + eps, max(first.y, second.y) + eps );
    
    if( point.x > upp_right_corner.x )
        return true;
    
    if( point.x < btm_left_corner.x )
        return true;
    
    if( point.y > upp_right_corner.y )
        return true;
    
    if( point.y < btm_left_corner.y )
        return true;

    return false;
}

double Boundary_Recall (vector<Point2d> boundary, vector<Line>& lines, double eps, vector<int>& boundary_found)
{
    double nb_found_boundary_points = 0;
    double boundary_recall = 0.0;
    map<int, vector<int32_t>> equations_parameters;
    
    Find_Line_Equation(lines, equations_parameters);
    
    for ( vector<Point2d>::iterator b_it = boundary.begin(), b_end=boundary.end(); b_it!=b_end; ++b_it )
    {
        for ( multimap<int, vector<int32_t>>::iterator it=equations_parameters.begin(); it!=equations_parameters.end(); ++it )
        {
            // Check if outside boundary point
            if( Check_If_Outside_Boundary_Box( *b_it, lines[(*it).first].first, lines[(*it).first].second, eps ) )
            {
                continue;
            }
            
            double dist_to_line = Compute_Distance_To_Line( *b_it , (*it).second );
            
            if( dist_to_line <= eps )
            {
                nb_found_boundary_points++;
                boundary_found.push_back((int)std::distance(boundary.begin(), b_it));
                break;
            }
        }
    }
    
    cout << "\nNumber of found boundary points: " << nb_found_boundary_points << "; All boundary points: " << boundary.size() << endl;
    
    boundary_recall = nb_found_boundary_points / boundary.size(); // Boundary recall
    
    cout << "Boundary recall: " << boundary_recall << endl;
    
    return boundary_recall;
}

// Read list of images names and sizes from text file
bool Read_Image_Names_Sizes (std::string const& file_name, std::vector<std::string>& image_names, std::vector<Point>& image_sizes)
{
    std::ifstream file;
    
    file.open( file_name );
    
    image_names.clear();
    
    image_sizes.clear();
    
    if ( ! file.is_open() ) { std::cout << "\nError opening " << file_name; return false; }
    
    int x, y;
    
    std::string line, name;
    
    while( getline( file, line ) )  // read one line from ifs
    {
        
        std::istringstream iss( line ); // access line as a stream
        
        iss >> name >> x >> y;
        
        image_names.push_back( name );
        
        image_sizes.push_back( Point2i( x, y ) );
    }
    
    return true;
}

//function which scans over the image and ouputs the image as a double array
double* ScanImageToDoubleArray (Mat& I)
{
    // accept only char type matrices (grayscale images)
    CV_Assert(I.depth() != sizeof(uchar));
    int channels = I.channels();
    
    int nRows = I.rows;
    int nCols = I.cols * channels;
    
    if (I.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }
    
    double * dI = (double *)malloc(I.rows * I.cols * sizeof(double));
    int i, j;
    uchar* p;
    
    for (i = 0; i < nRows; ++i)
    {
        p = I.ptr<uchar>(i);
        
        for (j = 0; j < nCols; ++j)
        {
            dI[j + i*I.cols] = p[j];
        }
    }
    
    return dI;
}

void Select_Boudnary_Points (string name, vector<Point2d>& boundary, int segmentation_nb, string path_to_dat_files)
{
    string path = path_to_dat_files + name + "_" + to_string( segmentation_nb ) + ".dat";
    Read_BSD_Human ( path, boundary );
}

double Measure_Boundary_Recall (string name, vector<Line>& lines, double eps, int num_segment, vector<Point2d>& boundary, vector<int>& boundary_found)
{
    bool debug = true;

    double b_recall = 0.0;
    
    b_recall = Boundary_Recall(boundary, lines, eps, boundary_found);
    
    if (debug) cout << "\nBoundary recall for: " + name + ", nb = " + to_string(num_segment) + " " + "is " + to_string(b_recall) << endl;
    
    return b_recall;
}

// Testing PLSD method
void Run_PLSD (string name, cv::Mat_<cv::Vec3b>& input_mat, std::vector<Line>& lines, int num_lines, double eps, int offset, Mat &fimage, float& exec_time, double sensitivity, bool gray_scale )
{
    cv::Mat greyMat;
    Mat_<Vec3b> colorMat;
    
    if(gray_scale)
    {
        cv::cvtColor(input_mat, greyMat, cv::COLOR_BGR2GRAY);
        cv::cvtColor(greyMat, colorMat, cv::COLOR_GRAY2RGB);
        input_mat = colorMat.clone();
    }
    
    // Measure execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    Find_Lines( input_mat, lines, num_lines, offset, fimage, sensitivity );
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto dur = t2 - t1;
    auto f_secs = std::chrono::duration_cast<std::chrono::duration<float>>(dur);
    exec_time = f_secs.count();
}

// Testing LSD method
void Run_LSD (string name, Mat& image, std::vector<Line>& lines_LSD, int& num_lines, double eps, Mat &fimage, float& exec_time)
{
    cv::Mat greyMat, colorMat;
    cv::cvtColor(image, greyMat, cv::COLOR_BGR2GRAY);
    
    // convert Mat image into an array of double values
    double * dImage = ScanImageToDoubleArray( greyMat );
    
    // Measure execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    // run LSD
    double * cloud = lsd( &num_lines, dImage, image.cols, image.rows ); // from lsd.c
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    auto dur = t2 - t1;
    auto f_secs = std::chrono::duration_cast<std::chrono::duration<float>>(dur);
    exec_time = f_secs.count();
    
    for (int i=0; i < num_lines; i++)
    {
        Point2d p1( cloud[7 * i + 0], cloud[7 * i + 1] );
        Point2d p2( cloud[7 * i + 2], cloud[7 * i + 3] );
        
        Line l = Line();
        l.first = p1;
        l.second = p2;
        
        lines_LSD.push_back( l );
    }
    
    fimage = image.clone();
}

void Save_Performance_To_Txt_File (string name, float num_lines_lsd, float num_lines_plsd, double LSD_BR2, float LSD_time, double PLSD_BR2, float PLSD_time, float nb_intersections, string path, int offset, double sensitivity)
{
    std::ostringstream oss;
    oss << to_string(sensitivity) + "_sensitivity_" + to_string(offset) + "_offset_all_BSD500_results" << ".txt";
    
    cout << oss.str() << endl;
    
    std::string file_name = oss.str();
    
    std::ofstream ofs;
    ofs.open (path + file_name, std::ofstream::out | std::ofstream::app);
    
    ofs << name << " " << num_lines_lsd << " " << nb_intersections << " " << LSD_BR2 << " " << LSD_time << " " << num_lines_plsd << " " << PLSD_BR2 << " " << PLSD_time << endl;
    
    ofs.close();
}

void Calculate_Average_Performance (string name, std::vector<float>& list_of_nb_lines_lsd, std::vector<float>& list_of_nb_lines_plsd, vector<double>& best_boudnary_recall_LSD, vector<double>& best_boudnary_recall_PLSD, vector<float>& LSD_time, vector<float>& PLSD_time, vector<float>& list_nb_intersections, string path, int offset, double sensitivity)
{
    float average_number_of_lines_lsd = std::accumulate( list_of_nb_lines_lsd.begin(), list_of_nb_lines_lsd.end(), 0.0) / list_of_nb_lines_lsd.size();
    
    float average_number_of_lines_plsd = std::accumulate( list_of_nb_lines_plsd.begin(), list_of_nb_lines_plsd.end(), 0.0) / list_of_nb_lines_plsd.size();
    
    float average_number_of_line_intersections_lsd = std::accumulate( list_nb_intersections.begin(), list_nb_intersections.end(), 0.0) / list_nb_intersections.size();
    
    double average_br_LSD = std::accumulate( best_boudnary_recall_LSD.begin(), best_boudnary_recall_LSD.end(), 0.0) / best_boudnary_recall_LSD.size();
    
    double average_br_PLSD = std::accumulate( best_boudnary_recall_PLSD.begin(), best_boudnary_recall_PLSD.end(), 0.0) / best_boudnary_recall_PLSD.size();
    
    float average_LSD_time = std::accumulate( LSD_time.begin(), LSD_time.end(), 0.0) / LSD_time.size();

    float average_PLSD_time = std::accumulate( PLSD_time.begin(), PLSD_time.end(), 0.0) / PLSD_time.size();
    
    Save_Performance_To_Txt_File( name, average_number_of_lines_lsd, average_number_of_lines_plsd, average_br_LSD, average_LSD_time, average_br_PLSD, average_PLSD_time, average_number_of_line_intersections_lsd, path, offset, sensitivity );
}

void Scale_Up_Image (Mat& src_image, Mat& dst_image, int scale)
{
    int y_size = src_image.cols;
    int x_size = src_image.rows;
    
    Size size = Size(y_size * scale, x_size * scale);
    resize(src_image, dst_image, size);//resize image
}

void Save_Boundary_Points (vector<Point2d>& boundary, cv::Mat& image, string const& name, string path_to_output_folder, int nb_segment, vector<int>&boundary_found, int scale)
{
    //All boundary points (Red).
    for ( int k = 0; k < boundary.size(); k++ )
    {
        image.at<cv::Vec3b>( boundary[k].y, boundary[k].x )[0] = 0;
        image.at<cv::Vec3b>( boundary[k].y, boundary[k].x )[1] = 0;
        image.at<cv::Vec3b>( boundary[k].y, boundary[k].x )[2] = 255;
    }
    
    //Found boundary points (Black).
    for ( int k = 0; k < boundary_found.size(); k++ )
    {
        image.at<cv::Vec3b>( boundary[boundary_found[k]].y, boundary[boundary_found[k]].x )[0] = 0;
        image.at<cv::Vec3b>( boundary[boundary_found[k]].y, boundary[boundary_found[k]].x )[1] = 0;
        image.at<cv::Vec3b>( boundary[boundary_found[k]].y, boundary[boundary_found[k]].x )[2] = 0;
    }
    
    Mat dst_image;
    Scale_Up_Image( image, dst_image, scale );
    
    Write_Image( dst_image, path_to_output_folder + name + "_h_" + to_string(nb_segment) + ".png" );
}

void Draw_Lines_With_Thickness (Mat& img, vector<Line>& lines)
{
    double percentage = 0.25;
    
    int nb_lines = (int)lines.size();
    
    int first_part = percentage * nb_lines;
    for ( int k = 0; k < first_part; k++ )
        line( img, lines[ k ].first, lines[ k ].second, Red, 4 );
    
    int second_part = first_part*2;
    for ( int k = first_part; k < second_part; k++ )
        line( img, lines[ k ].first, lines[ k ].second, Red, 3 );
    
    int third_part = first_part*3;
    for ( int k = second_part; k < third_part; k++ )
        line( img, lines[ k ].first, lines[ k ].second, Red, 2 );
    
    int fourth_part = first_part*4;
    for ( int k = third_part; k < fourth_part; k++ )
        line( img, lines[ k ].first, lines[ k ].second, Red, 1 );
}

int Number_Intersections_LSD (vector<Line>& lines)
{
    int nb_inters = 0;
    
    //Find parameters for each line
    map<int, vector<int32_t>> equations_parameters;
    Find_Line_Equation(lines, equations_parameters);
    
    for ( multimap<int, vector<int32_t>>::iterator it1=equations_parameters.begin(); it1!=equations_parameters.end(); ++it1 )
    {
        int32_t a1 = (*it1).second[0];
        int32_t b1 = (*it1).second[1];
        int32_t c1 = (*it1).second[2];
        
        int ind1 = (*it1).first;
        int f_x1 = lines[ind1].first.x;
        int f_y1 = lines[ind1].first.y;
        int s_x1 = lines[ind1].second.x;
        int s_y1 = lines[ind1].second.y;
        
        for ( multimap<int, vector<int32_t>>::iterator it2=it1; it2!=equations_parameters.end(); ++it2 )
        {
            int32_t a2 = (*it2).second[0];
            int32_t b2 = (*it2).second[1];
            int32_t c2 = (*it2).second[2];
            
            int ind2 = (*it2).first;
            int f_x2 = lines[ind2].first.x;
            int f_y2 = lines[ind2].first.y;
            int s_x2 = lines[ind2].second.x;
            int s_y2 = lines[ind2].second.y;
            
            // both endpoints of the 1st segment should be on different sides of the 2nd line
            int32_t s_line1 = a2*f_x1 + b2*f_y1 + c2;
            int32_t s_line2 = a2*s_x1 + b2*s_y1 + c2;
            
            // both endpoints of the 2st segment should be on different sides of the 1st line
            int32_t f_line1 = a1*f_x2 + b1*f_y2 + c1;
            int32_t f_line2 = a1*s_x2 + b1*s_y2 + c1;
            
            if ( ( ( s_line1 )*( s_line2 ) < 0) and ( ( ( f_line1 )*( f_line2 ) < 0 ) ) )
            {
                nb_inters++;
            }
        }
    }
    
    return nb_inters;
}

void Find_Difference (double BR2_PLSD, double BR2_LSD, string id, multimap <double, string>& m)
{
    double diff = abs( BR2_PLSD - BR2_LSD );
    
    m.insert( pair<double, string> (diff, id) );
}

void Highest_Difference (multimap <double, string>& m, int nb_top_images, vector<string>& top_images)
{
    for (std::multimap<double,string>::iterator it=m.begin(); it!=m.end(); ++it)
    {
        int dist = (int)std::distance(m.begin(), it);
        
        if( dist < nb_top_images)
        {
            top_images.push_back( (*it).second );
            std::cout << ' ' << (*it).second;
        }
        else break;
    } 
}

void Save_Top_Images (vector<string>& top_images, int offset, double sensitivity, string path)
{
    std::ostringstream oss;
    oss << to_string(sensitivity) + "_sensitivity_" + to_string(offset) + "_offset_10_top_results" << ".txt";
    
    cout << oss.str() << endl;
    
    std::string file_name = oss.str();
    
    std::ofstream ofs;
    ofs.open (path + file_name, std::ofstream::out | std::ofstream::app);
    
    for (std::vector<string>::iterator it = top_images.begin() ; it != top_images.end(); ++it)
    {
        ofs << *it << endl;
    }
    
    ofs.close();
}

// Accuracy
double Compute_ACC (double TP, double TP_FP, double TN, double FN)
{
    double ACC = (TP + TN) / (TP_FP + FN + TN);
    
    return ACC;
}

// Sensitivity
double Compute_True_Positive_Rate (double TP, double FN)
{
    double TPR = TP / (TP + FN);
    
    return TPR;
}

//Precision
double Compute_Positive_Predictive_Value (double TP, double TP_FP)
{
    double PPV = TP / TP_FP;

    return PPV;
}

void Discretize_Segments (vector<Line>& lines, vector<Point2d>& output, Point& sizes)
{
    Mat img( Mat( sizes.y, sizes.x, CV_8UC3 ) );

    img = White;
    
    // Draw lines
    for ( int k = 0; k < lines.size(); k++ )
        line( img, lines[ k ].first, lines[ k ].second, Black, 1 );
    
    // Extract black pixels
    for (int i = 0; i < img.cols; i++ )
    {
        for (int j = 0; j < img.rows; j++)
        {
             if (img.at<uchar>(j, i) == 0)
             {
                 output.push_back(Point(j,i));
             }
        }
    }
    
    cout << "\nNb of pixels in image: " << sizes.x * sizes.y << endl;
    cout << "Nb of black pixels: " << (int)output.size() << endl;
}

double Compute_TP (vector<Point2d>& black_px, vector<Point2d>& boundary)
{
    double TP = 0.0;
    
    for (int i=0; i<(int)boundary.size(); ++i)
    {
        for (int j=0; j<(int)black_px.size(); ++j)
        {
            if(black_px[j].x == (int)boundary[i].x and black_px[j].y == (int)boundary[i].y)
                TP++;
        }
    }
    
    return TP;
}

void Save_Statistical_Measures_To_Txt (string fname, string name, string path, double tpr, double ppv, double acc)
{
    std::ostringstream oss;
    oss << fname << "_statistics" << ".txt";
    
    cout << oss.str() << endl;
    
    std::string file_name = oss.str();
    
    std::ofstream ofs;
    ofs.open (path + file_name, std::ofstream::out | std::ofstream::app);
    
    ofs << name << " " << tpr << " " << ppv << " " << acc << endl;
    
    ofs.close();
}

void Compute_Statistical_Measures (Point& sizes, vector<Point2d>& black_px, vector<Point2d>& boundary, string name, string path, string fname)
{
    bool debug = true;
    int nb_black_px = (int)black_px.size();
    int nb_gt = (int)boundary.size();
    
    double TP = Compute_TP(black_px, boundary);
    double TPR = Compute_True_Positive_Rate(TP, nb_gt);
    
    if(debug) cout << "\nTPR=" << TPR << endl;
    
    double PPV = Compute_Positive_Predictive_Value(TP, nb_black_px);
    
    if(debug) cout << "\nPPV=" << PPV << endl;

    double TN = sizes.x*sizes.y - (nb_gt + nb_black_px);
    double FN = nb_gt - TP;
    double TP_FP = nb_black_px;
    
    double ACC = Compute_ACC(TP, TP_FP, TN, FN);
    
    if(debug) cout << "\nACC=" << ACC << endl;
    
    // Save results to txt
    Save_Statistical_Measures_To_Txt(fname, name, path, TPR, PPV, ACC);
}

int main(int argc, const char * argv[])
{
    cout << "Persistent Line Detector\n";
    
    bool debug = true;
    bool save_images = false; //flag to save PLSD and LSD output as images
    bool save_averages = false; //flag to calculate and save averages
    bool calculate_br2 = true;
    bool gray_scale = false; //if we want to process grayscale images
    bool scale_up_image = false;

    // Number of images for testing
    int num_images = 5;
    
    // Number of lines segments
    int num_lines = 0;
    
    // Offset parameter
    double eps = 2.0;
    
    // Offset parameter for PLSD algorithm
    int offset = 5;
    
    // Sensitivity for threold
    double sensitivity = 0.60;
    
    // Nb of intersections for LSD algorithm
    int nb_intersections = 0;
    
    // Scale parameter for images
    int scale_up_param = 4;
    
    int max_nb_of_segmentations = 0;
    
    //Name of one testing image
    string name = "8023"; //"61034"; //"Testing_Image";
    
    // File extension
    string ext = "jpg";//"png";
    
    // Mats for saving images
    Mat img_LSD;
    Mat img_PLSD;
    
    vector<std::string> image_names;
    vector<Point> image_sizes;
    
    vector<double> best_boudnary_recall_LSD;
    vector<double> best_boudnary_recall_PLSD;
    
    vector<Line> lines_LSD;
    vector<Line> lines_PLSD;
    vector<float> list_of_nb_lines_LSD;
    vector<float> list_of_nb_lines_PLSD;
    vector<float> list_nb_intersections_LSD;

    vector<float> LSD_time;
    vector<float> PLSD_time;

    string path_folder_to_list_of_images = "/Users/Grzegorz/Desktop/project_internship/Persistent_Line_Detector/"; // path to BSD500sizes.txt
    string path_to_BSD500_folder = "/Users/Grzegorz/Desktop/project_internship/BSR/BSDS500/data/images/all/";   // path to all 500 images from BSD500
    string path_to_output_folder = "/Users/Grzegorz/Desktop/project_internship/Persistent_Line_Detector/output_results/results_for_updated_code/pairs_of_good_images/"; //path to output dir of both algorithms
    string path_to_dat_files = "/Users/Grzegorz/Desktop/project_internship/Persistent_Line_Detector/data/";
    
    Read_Image_Names_Sizes( path_folder_to_list_of_images + "BSD500sizes.txt", image_names, image_sizes );
    
    if ( num_images == 1 ) debug = true;
    
    if (debug) cout << "Nb of analyzed images: " << num_images << endl;
    
    multimap <double, string> list_of_perf_differences;
    
    // Loop over chosen number of images from BSD500
    for ( int i = 0; i < num_images; i++ )
    {
        lines_LSD.clear();
        lines_PLSD.clear();
        num_lines = 0;
        max_nb_of_segmentations = 0;
        
        if ( num_images > 1 ) name = image_names[i];
        
        if (debug) cout << "i=" << i << " File name: " << name << endl;
        
        string input_file = path_to_BSD500_folder + name + "." + ext;
        Mat_<Vec3b> input_color, input_mat;
        Point sizes;
        
        Load_Image( input_file, sizes, input_color, input_mat );
        
        float exec_time_lsd = 0.0;
        float exec_time_plsd = 0.0;
        
        map<double, int> list_of_boundary_recalls_LSD;
        map<double, int> list_of_boundary_recalls_PLSD;
        
        //Run Line Segment Detector
        Run_LSD( name, input_mat, lines_LSD, num_lines, eps, img_LSD, exec_time_lsd );
        
        //number of segment intersections
        nb_intersections = Number_Intersections_LSD( lines_LSD );
        list_nb_intersections_LSD.push_back(nb_intersections);
        
        //Run Persistent Line Segment Detector
        Run_PLSD( name, input_mat, lines_PLSD, num_lines, eps, offset, img_PLSD, exec_time_plsd, sensitivity, gray_scale );
        
        vector<Point2d> PLSD_discretised_segments;
        Discretize_Segments( lines_PLSD, PLSD_discretised_segments, sizes );
        
        vector<Point2d> LSD_discretised_segments;
        Discretize_Segments( lines_LSD, LSD_discretised_segments, sizes );
        
        cout << "Nb lines: " << endl;
        cout << "LSD: " << num_lines << endl;
        cout << "PLSD: " << lines_PLSD.size() << endl;
        
        if(debug) cout << exec_time_lsd << " " << exec_time_plsd << endl;
        
        LSD_time.push_back(exec_time_lsd);
        
        PLSD_time.push_back(exec_time_plsd);
        
        vector<Point2d> boundary;

        if( calculate_br2 )
        {
        // Find number of human segmentations for a given input image
        max_nb_of_segmentations = Find_File_Names(name, path_to_dat_files);
        
        for (int segment_nb = 1; segment_nb <= max_nb_of_segmentations; ++segment_nb)
        {
            boundary.clear();
            
            Select_Boudnary_Points( name, boundary, segment_nb, path_to_dat_files );
        
            vector<int> boundary_found;
            
            // Measure boundary recall for LSD
            double br_lsd = Measure_Boundary_Recall( name, lines_LSD, eps, segment_nb, boundary, boundary_found );
            list_of_boundary_recalls_LSD.insert( make_pair(br_lsd, segment_nb) );
            
            if( save_images )
            {
                Mat img_bcgd_lsd( Mat( sizes.y, sizes.x, CV_8UC3 ) );
                img_bcgd_lsd = White;
                Save_Boundary_Points( boundary, img_bcgd_lsd, name + "_LSD", path_to_output_folder, segment_nb, boundary_found, scale_up_param );
            }
            
            boundary_found.clear();
            
            double br_plsd = Measure_Boundary_Recall( name, lines_PLSD, eps, segment_nb, boundary, boundary_found );
            list_of_boundary_recalls_PLSD.insert( make_pair(br_plsd, segment_nb) );
            
            if( save_images )
            {
                Mat img_bcgd_plsd( Mat( sizes.y, sizes.x, CV_8UC3 ) );
                img_bcgd_plsd = White;
                Save_Boundary_Points( boundary, img_bcgd_plsd, name + "_PLSD", path_to_output_folder, segment_nb, boundary_found, scale_up_param );
            }
        }
        
        // Return the best boundary recall for a given image (LSD)
        double best_br_lsd = list_of_boundary_recalls_LSD.rbegin()->first;
        if (debug) cout<<"\nBest boundary recall for LSD: "<< best_br_lsd << "\tNr: " << list_of_boundary_recalls_LSD.rbegin()->second << endl;
        best_boudnary_recall_LSD.push_back( best_br_lsd );
            
        // Return the best boundary recall for a given image (PLSD)
        double best_br_plsd = list_of_boundary_recalls_PLSD.rbegin()->first;
        if (debug) cout<<"\nBest boundary recall for PLSD: "<< best_br_plsd << "\tNr: " << list_of_boundary_recalls_PLSD.rbegin()->second << endl;
        best_boudnary_recall_PLSD.push_back( best_br_plsd );
            
        // Compute sensitivity, precision, accuracy
        boundary.clear();
        Select_Boudnary_Points( name, boundary, list_of_boundary_recalls_PLSD.rbegin()->second, path_to_dat_files );
        Compute_Statistical_Measures( sizes, PLSD_discretised_segments, boundary, name, path_to_output_folder, "PLSD" );
        
        // Find differences in BR2
        Find_Difference(best_br_plsd, best_br_lsd, name, list_of_perf_differences);
            
        list_of_nb_lines_LSD.push_back( num_lines );
        list_of_nb_lines_PLSD.push_back( (int)lines_PLSD.size() );
        
        Save_Performance_To_Txt_File( name, num_lines, (int)lines_PLSD.size(), best_br_lsd, exec_time_lsd, best_br_plsd, exec_time_plsd, nb_intersections, path_to_output_folder, offset, sensitivity );
        
        list_of_boundary_recalls_LSD.clear();
        list_of_boundary_recalls_PLSD.clear();
        }
        
        if( save_images )
        {
            Mat dst_image_LSD;
            if(scale_up_image) Scale_Up_Image( img_LSD, dst_image_LSD, scale_up_param );
            else
                dst_image_LSD = img_LSD.clone();
            
            Draw_Lines( dst_image_LSD, lines_LSD );
            //Draw_Lines_With_Thickness( img_LSD, lines_LSD );
            
            Write_Image( dst_image_LSD, path_to_output_folder + name+ "_LSD" + ".png" );
        
            Mat dst_image_PLSD;
            if(scale_up_image) Scale_Up_Image( img_PLSD, dst_image_PLSD, scale_up_param );
            else
                dst_image_PLSD = img_PLSD.clone();
            
            Draw_Lines( dst_image_PLSD, lines_PLSD );
            //Draw_Lines_With_Thickness( img_PLSD, lines_PLSD );
            
            Write_Image( dst_image_PLSD, path_to_output_folder + name + "_PLSD" + ".png" );
        }
    }
    
    vector<string> top_images;
    
    // Find ten top images
    Highest_Difference( list_of_perf_differences, 10, top_images );
    
    //Save ids to txt file
    Save_Top_Images( top_images, offset, sensitivity, path_to_output_folder );
    
    // Computer average performance
    if(save_averages)
        Calculate_Average_Performance ( "average:", list_of_nb_lines_LSD, list_of_nb_lines_PLSD, best_boudnary_recall_LSD, best_boudnary_recall_PLSD, LSD_time, PLSD_time, list_nb_intersections_LSD, path_to_output_folder, offset, sensitivity );
    
    return 0;
}

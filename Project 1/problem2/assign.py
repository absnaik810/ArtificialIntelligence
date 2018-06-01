#!/usr/bin/env python
'''
To solve this question, we envisaged the problem to be related to n-queens.  Just like in n-queens, we first try to meet one of the
constraints (the heavier ones), then the next and so on.  Also, we assume throughout that k>m>n. The problem formulation is as below:

Initial state:  We consider each student to be a single group.  Thus, if there are n students, then there are n groups.

State space:  The state space would be the combination of all the students to form a team in various ways, such that the
team size is not more than 3.

Successor function:  We consider the student with the highest cost and try to pair him with his
preferences.  So, at any instant, the student being worked upon is the one who has the maximum cost.

Edge weights: Each group costs k minutes, each student takes a single minute if he is allocated a team size that he didn't prefer,
`n` minutes per person if assigned to someone they didn't request and `m` minutes per person if assigned to someone they do not want to work with.

Goal state: A set of groups of students such that their total cost would be lesser than that of the threshold.

Working of the search algorithm:
    The algorithm initially treats each student to be an individual group.  Thus, if there are `n` students, then there are `n` groups.
Next, it finds out the student(s) with the maximum cost and tries to form groups as per his preferences.  While doing this, it does not bother
if other students want to (or do not want to) work with this student because we assume that k>m>n.  Once this group has been formed, it calculates
the total cost.  If this total cost is less than the threshold (initially set equal to 1), then it simply exits; if not then it increments the
threshold by some pre-defined value (say 10) and then again calculates the cost for each student.  Note that the value of `k` is not used in this
case (and it is used later on after forming the final groups).  Once the costs for each student have been found, it again selects the highest
student and starts forming groups as per his preference.  Then it again calculates the total cost to find if it is less than the threshold, if
yes then exits; if not then increments the threshold and iterates.  It finally exits the iteration when the total cost it found is lower than
the threshold in that case.

    Once the loop is exited, it finds to see if there exists any student (like Steflee) who has not been paired up with.  If yes, then it pairs
such a student with some other group (provided the group size<=2).  It then finally calculates the total cost (also taking into account `k`
this time).  The groups thus formed are displayed alongwith the total cost.

    This approach works and generates the lowest possible cost because we increment the threshold in each iteration.  Thus, the threshold
acts as an upper limit to the minimum cost that it needs to find.  It uses a hardThreshold set to a very high value to ensure that the search
does not run to infinity.

Assumptions and challenges faced:
    We assume that k>m>n.  Thus, throughout the code, we try to maximize the number of groups and to pair up the students with their preferences.
The main challenge we faced was trying to determine a goal state (since the program could literally find a max value every time and keep
iterating).  It is then that we decided to come up with the threshold factor to ensure that we have an upper bound on the minimum cost that
is found.  Besides this, we also have a hardThreshold set to a very high value to ensure that the search doesn't run to infinity.  We also
initially thought about the brute force approach of pair students in all possible ways, but quickly determined that it would be very expensive
and might not run even for relatively small number of students.  There after we envisaged it to be similar to n-queens problem and
employed a similar thought process for solving it.
'''

import sys
import copy as cp
# get the parameters from the command line
fileName = sys.argv[1]
k = int(sys.argv[2])
m = int(sys.argv[3])
n = int(sys.argv[4])

fileData = open(fileName, "r")
data = fileData.read()

# convert data into a list (separated by a ',')
data = data.rstrip('\n').split("\n")

studentCosts = {}
survey = {}
for element in data:
    '''
    This loop is to create a dictionary to represent the input format in a better way for processing it
    The key would be the user ID
    'rank' represents whether the user prefers to work alone, in a group of two or three or he has no preference
    'preferences' represents the students the current user (student) prefers to work with
    'notWannaWork' represents the students the current user (student) does not prefer to work with
    '''
    element = element.split()
    element[2] = [items for items in element[2].split(',')]
    element[3] = [items for items in element[3].split(',')]
    survey.update({element[0]: {'rank': int(element[1]), 'preferences': element[2], 'notWannaWork': element[3]}})

def getMax(studentCosts):
    '''
    Given a dictionary as the input, it iterates over the <key, value> pairs and returns the maximum value
    '''
    myMax = -1
    for key in studentCosts:
        if int(studentCosts[key]) > myMax:
            myMax = studentCosts[key]
    return myMax

def getM(value, group, survey):
    '''
    getM() works as follows:
    For each person who has been teamed up with the current student (value), if that person is in his list of notWannaWork then
    increment the value of `val` by `m`
    Finally return `val`
    '''
    val = 0
    for each in group[value]['team']:
        if each in survey[value]['notWannaWork']:
            val += m
    return val

def getK(value, group, survey):
    '''
    getK() works as follows:
    it returns the value of `k`
    Also, technically this function can be skipped; but it is kept as is, as a good coding practice
    '''
    return k

def getN(value, group, survey):
    '''
    getN() works as follows:
    For each student who the current student wants to work with (preferences) but has not been teamed up with (team), increment
    the value of `val` by `n`
    Finally return `val`
    '''
    val = 0
    temp = []
    temp = survey[value]['preferences']
    teamList = []
    teamList = group[value]['team']
    for each in temp:
        if each == '_':
            continue
        if each not in teamList:
            val += n
    return val

def shouldAddOne(value, group, survey):
    '''
    shouldAddOne() works as follows:
    If the size of the team with which the current student wants to work with is not equal to the team size that he has been allocated to,
    then add 1 (since he will spend 1 minute only)
    Return 1 or 0
    '''
    if survey[value]['rank'] == 0:
        return 0
    count = 0
    temp = []
    temp = group[value]['team']
    if len(temp) != survey[value]['rank']:
        return 1
    else:
        return 0

def get_currCost(studentCosts):
    '''
    This function is used to calculate the cost, given a dictionary as an input
    '''
    value = 0
    for key in studentCosts:
        value += studentCosts[key]
    return value

def update_cost(finalGroup):
    '''
    As the name suggests, the update_cost() function is used to update the cost after the students are grouped.
    It calculates the cost for each student and returns the values as a dictionary
    '''
    temp = 0;
    studentCosts = {}
    for each in finalGroup:
        temp = getM(each, finalGroup, survey) + getN(each, finalGroup, survey) + shouldAddOne(each, finalGroup, survey)
        studentCosts[each] = temp
    return studentCosts

def make_groups(group, studentCosts):
    '''
    As the name suggests, the make_groups() function is used to generate groups.  It works on the student with the maximum current cost
    and tries to allocate groups as per his preference.  It also ensures that the same student is not added twice in a group.
    '''
    student = max(studentCosts, key=studentCosts.get)
    tmp_student = studentCosts
    foundMax = False
    while not foundMax:
        if survey[student]['preferences'][0] == '_':
            s = tmp_student.pop(student)
            student = max(tmp_student, key=studentCosts.get)
        else:
            foundMax = True
    templist = []
    templist.append(student)
    # add each student in djcran's preference to his team
    for each in survey[student]['preferences']:
        if each == '_':
            continue
        # below if condition ensures that the same person is not added twice
        if len(group[each]['team']) > 1:
            tmp_group = cp.deepcopy(group)
            prev_team = tmp_group[each]['team']
            prev_team.pop(prev_team.index(each))
            stud_tmp_list = cp.deepcopy(templist)
            stud_tmp_list.append(each)
            tmp_group[student]['team'] = stud_tmp_list
            tmp_group[each]['team'] = stud_tmp_list
            tmp_cost = update_cost(tmp_group)
            curr_cost = update_cost(group)
            t_cost = tmp_cost[student] + tmp_cost[each]
            c_cost = curr_cost[student] + curr_cost[each]
            for stud in prev_team:
                t_cost += tmp_cost[stud]
                c_cost += curr_cost[stud]
            if t_cost < c_cost:
                templist.append(each)
                group = tmp_group
                group[student]['team'] = templist
                group[each]['team'] = templist
            continue
        templist.append(each)
        group[student]['team'] = templist
        group[each]['team'] = templist
    return group

group = {}
for value in survey:
    # To start with, we treat each student as a single group.  This is our initial state
    group.update({value: {'rank': [], 'preferences': [], 'notWannaWork': [], 'team': []}})
    # Note that getK() is not being called here since we would be working on the basis of individuals.  getK() would be called later on,
    # when we have to determine the final cost
    finalValue = shouldAddOne(value, group, survey) + getM(value, group, survey) + getN(value, group, survey)
    # updating the studentCosts dictionary
    studentCosts.update({value: finalValue})

# setting the initial threshold as `1`
threshold = 1
hardThreshold = 9999999999
finalGroup = group
newStudentCosts = studentCosts

while True:
    val = getMax(newStudentCosts)
    finalGroup = make_groups(finalGroup, newStudentCosts)
    newStudentCosts = update_cost(finalGroup)
    currCost = get_currCost(newStudentCosts)
    if currCost < threshold:
        break
    # increment the threshold by 10 on each iteration
    # by means of a threshold, we plan to set an upperbound on the time for which a program can run
    threshold += 10
    # just to ensure that the program doesn't run to infinity!
    if threshold>hardThreshold:
        break

# using a loop to detect if students without a group (like Steflee in the given example) can be (forcibly!) added to a group
# this makes sense since we assume that k>m>n, which means it would be costly to treat a single student like Steflee as a team
for each in finalGroup:
    if not finalGroup[each]['team'] or (len(finalGroup[each]['team'])==1 and finalGroup[each]['team'][0]==each):
        for everyPreference in survey[each]['preferences']:
            if everyPreference == '_':
                continue
            #check if the team size of any student he wants to work with<3.  If yes, then assign him to it.
            if len(finalGroup[everyPreference]['team']) < 3:
                finalGroup[everyPreference]['team'].append(each)
                finalGroup[each]['team'] = finalGroup[everyPreference]['team']
                break
        for eachValue in finalGroup:
            if each == eachValue:
                continue
            if eachValue == '_':
                continue
            #check if any of the groups have a size<3, if yes, then simply assign him to it.
            if not finalGroup[each]['team'] and len(finalGroup[eachValue]['team']) < 3:
                finalGroup[eachValue]['team'].append(each)
                finalGroup[each]['team'] = finalGroup[eachValue]['team']
                break
newStudentCosts = update_cost(finalGroup)
currCost = get_currCost(newStudentCosts)

# use a set to store each of the teams formed.
# using a set will ensure uniqueness
# using a counter to count the number of students without any teams (ideally it should be just 1 because if there is more than 1, then they can be paired)
s = set()
counter = 0
for each in finalGroup:
    # check if his team is empty
    if not finalGroup[each]['team']:
        counter += 1
        continue
    # store each group in the set as a tuple
    t = tuple(finalGroup[each]['team'])
    s.add(t)

# print each group followed by the total cost
# note that we are using `k` here to print the total cost based on the number of groups
for each in s:
    print(" ".join(each))
print(k * (len(s) + counter) + currCost)
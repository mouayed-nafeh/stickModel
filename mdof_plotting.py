# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 17:00:26 2023

@author: Moayad
"""

def formatAx(axModel,Title,xLabel,yLabel,titleFontSize = 12, otherFontSize = 12):
    axModel.grid(True,color='grey')
    handles, labels = axModel.get_legend_handles_labels()
    if len(handles)>0:
        axModel.legend(fontsize=otherFontSize)
    axModel.set_title(Title, fontsize=titleFontSize)
    axModel.set_xlabel(xLabel, fontsize=otherFontSize)
    axModel.set_ylabel(yLabel, fontsize=otherFontSize)            
    axModel.tick_params('x', labelsize=otherFontSize, rotation=0)
    axModel.tick_params('y', labelsize=otherFontSize, rotation=0)
    
def set_aspect_equal_3d(ax):
    #"""Fix equal aspect bug for 3D plots."""
    #https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    import matplotlib.pyplot as plt

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius]) 

def plot_model_nodes(display_info):

    # import necessary libraries
    import matplotlib.pyplot as plt
    import openseespy.opensees as ops

    # get list of model nodes
    NodeCoordListX = []; NodeCoordListY = []; NodeCoordListZ = [];
    NodeMassList = []
    
    nodeList = ops.getNodeTags()
    for thisNodeTag in nodeList:
        NodeCoordListX.append(ops.nodeCoord(thisNodeTag,1))
        NodeCoordListY.append(ops.nodeCoord(thisNodeTag,2))
        NodeCoordListZ.append(ops.nodeCoord(thisNodeTag,3))
        NodeMassList.append(ops.nodeMass(thisNodeTag,1))
            
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(nodeList)):
        ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],s=50,color='black')
        if display_info == True:
            ax.text(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],  'Node#:%s (%s,%s,%s)' % (str(i),str(NodeCoordListX[i]),str(NodeCoordListY[i]),str(NodeCoordListZ[i])), size=20, zorder=1, color='black') 
        
    ax.set_xlabel('X Coordinates, [m]')
    ax.set_ylabel('Y Coordinates, [m]')
    ax.set_zlabel('Z Coordinates, [m]')
    
   # plt.show()    

def plot_model_elements(display_info):

    # import necessary libraries
    import matplotlib.pyplot as plt
    import openseespy.opensees as ops

    modelLineColor = 'blue'
    modellinewidth = 1
    Vert = 'Z'
   
    
    # get list of model nodes
    NodeCoordListX = []; NodeCoordListY = []; NodeCoordListZ = [];
    NodeMassList = []
    
    nodeList = ops.getNodeTags()
    for thisNodeTag in nodeList:
        NodeCoordListX.append(ops.nodeCoord(thisNodeTag,1))
        NodeCoordListY.append(ops.nodeCoord(thisNodeTag,2))
        NodeCoordListZ.append(ops.nodeCoord(thisNodeTag,3))
        NodeMassList.append(ops.nodeMass(thisNodeTag,1))
    
    # get list of model elements
    elementList = ops.getEleTags()
    for thisEleTag in elementList:
        eleNodesList = ops.eleNodes(thisEleTag)
        if len(eleNodesList)==2:
            [NodeItag,NodeJtag] = eleNodesList
            NodeCoordListI=ops.nodeCoord(NodeItag)
            NodeCoordListJ=ops.nodeCoord(NodeJtag)
            [NodeIxcoord,NodeIycoord,NodeIzcoord]=NodeCoordListI
            [NodeJxcoord,NodeJycoord,NodeJzcoord]=NodeCoordListJ

    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(nodeList)):
        ax.scatter(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],s=50,color='black')
        if display_info == True:
            ax.text(NodeCoordListX[i],NodeCoordListY[i],NodeCoordListZ[i],  'Node#:%s (%s,%s,%s)' % (str(i),str(NodeCoordListX[i]),str(NodeCoordListY[i]),str(NodeCoordListZ[i])), size=20, zorder=1, color='black') 
    
    i = 0
    while i < len(elementList):
        
        x = [NodeCoordListX[i], NodeCoordListX[i+1]]
        y = [NodeCoordListY[i], NodeCoordListY[i+1]]
        z = [NodeCoordListZ[i], NodeCoordListZ[i+1]]
        
        plt.plot(x,y,z,color='blue')
        i = i+1
    
    plt.show()
    
    
    
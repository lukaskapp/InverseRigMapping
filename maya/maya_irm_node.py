import maya.api.OpenMaya as om
import math, sys

def maya_useNewAPI():
    pass


class inverseRigMappingNode(om.MPxNode):
    node_id = om.MTypeId(0x7f001) # unique ID
    node_input = om.MObject()
    node_output = om.MObject()

    # method to return an instance
    @staticmethod
    def creator():
        return inverseRigMappingNode()


    # method called by Maya at initialization
    # set attributes
    @staticmethod
    def initialize():
        # attributes are defined using the create method of a subclass of the MFnAttribute class
        #nAttr = om.MFnNumericAttribute()
        #inverseRigMappingNode.input = nAttr.create("input", "i", om.MFnNumericData.kFloat, 0.0)
        #nAttr.storable = True
        #nAttr.writable = True

        
        # compound attribute
        """
        cAttr = om.MFnCompoundAttribute()
        uAttr = om.MFnUnitAttribute()
        inverseRigMappingNode.input = cAttr.create("input", "i")
        inverseRigMappingNode.inputX = uAttr.create("inputX", "iX", om.MFnUnitAttribute.kDistance)
        inverseRigMappingNode.inputY = uAttr.create("inputY", "iY", om.MFnUnitAttribute.kDistance)
        inverseRigMappingNode.inputZ = uAttr.create("inputZ", "iZ", om.MFnUnitAttribute.kDistance)
        cAttr.addChild(inverseRigMappingNode.inputX)
        cAttr.addChild(inverseRigMappingNode.inputY)
        cAttr.addChild(inverseRigMappingNode.inputZ)
        """

        


        nAttr = om.MFnNumericAttribute()
        inverseRigMappingNode.output = nAttr.create("output", "o", om.MFnNumericData.kFloat, 0.0)
        nAttr.readable = True
        nAttr.storable = False
        nAttr.writable = False

        # after defining, execute addAttribute of MPxNode
        inverseRigMappingNode.addAttribute(inverseRigMappingNode.input)
        inverseRigMappingNode.addAttribute(inverseRigMappingNode.output)

        # also set the output to be recalculated when the input is changed
        inverseRigMappingNode.attributeAffects(inverseRigMappingNode.input, inverseRigMappingNode.output)

    # constructor calls parent constructor
    def __init__(self):
        om.MPxNode.__init__(self)

    # a method called by Maya when the value of an attribute is calculated
    def compute(self, plug, datablock):
        if(plug == inverseRigMappingNode.output):
            dataHandle = datablock.inputValue(inverseRigMappingNode.input)
            inputFloat = dataHandle.asFloat()
            result = math.sin(inputFloat) * 10
            outputHandle = datablock.outputValue(inverseRigMappingNode.output)
            outputHandle.setFloat(result)
            datablock.setClean(plug)

# a function called by Maya that registers a new node
def initializePlugin(obj):
    mplugin = om.MFnPlugin(obj)

    try:
        mplugin.registerNode("inverseRigMappingNode", inverseRigMappingNode.node_id, inverseRigMappingNode.creator,
                                inverseRigMappingNode.initialize, om.MPxNode.kDependNode)
    except:
        sys.stderr.write("Failed to register node: {}".format("inverseRigMappingNode"))
        raise

# a function called by Maya when exiting the plug_in
def uninitializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(inverseRigMappingNode.node_id)
    except:
        sys.stderr.write("Failed to uninitialize node: {}".format("inverseRigMappingNode"))
        raise

"""
    def compute(self, plug, datablock):
        # get handles from MPxNode's data block
        aHandle = datablock.inputValue(inverseRigMapping.a)
        bHandle = datablock.inputValue(inverseRigMapping.b)
        resultHandle = datablock.outputValue(inverseRigMapping.result)

        # get data from handles
        a = aHandle.asFloat()
        b = bHandle.asFloat()

        # compute
        c = a + b

        # output
        resultHandle.setFloat(c)



def create():
    pass

def init():
    # setup input attributes
    nAttr = om.MFnNumericAttribute() # Maya's numeric attr class
    kFloat = om.MFnNumericData.kFloat # Maya's float type
    inverseRigMappingNode.a = nAttr.create("a", "a", kFloat, 0.0)
    nAttr.hidden = False
    nAttr.keyable = True
    inverseRigMappingNode.b = nAttr.create("b", "b", kFloat, 0.0)
    nAttr.hidden = False
    nAttr.keyable = True
    
    # setup the output attributes
    inverseRigMappingNode.result = nAttr.create("result", "r", kFloat)
    nAttr.writable = False
    nAttr.storable = False
    nAttr.readable = True

    # add attributes to node
    inverseRigMappingNode.addAttribute(inverseRigMappingNode.a)
    inverseRigMappingNode.addAttribute(inverseRigMappingNode.b)
    inverseRigMappingNode.addAttribute(inverseRigMappingNode.result)

    # set the attribute dependencies
    inverseRigMappingNode.attributeAffects(inverseRigMappingNode.a, inverseRigMappingNode.result)
    inverseRigMappingNode.attributeAffects(inverseRigMappingNode.b, inverseRigMappingNode.result)

def _topplugin(mobject):
    pass

def initializePlugin(mobject):
    pass

def uninitializePlugin(mobject):
    pass
"""
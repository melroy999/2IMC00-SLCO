/**
 */
package org.xtext.textualSlco.impl;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import org.xtext.textualSlco.ArgumentType;
import org.xtext.textualSlco.Assignment;
import org.xtext.textualSlco.BidirectionalChannel;
import org.xtext.textualSlco.BinaryOperatorExpression;
import org.xtext.textualSlco.BooleanConstantExpression;
import org.xtext.textualSlco.Channel;
import org.xtext.textualSlco.ChannelType;
import org.xtext.textualSlco.Delay;
import org.xtext.textualSlco.Expression;
import org.xtext.textualSlco.Final;
import org.xtext.textualSlco.Initial;
import org.xtext.textualSlco.IntegerConstantExpression;
import org.xtext.textualSlco.Model;
import org.xtext.textualSlco.Operator;
import org.xtext.textualSlco.Port;
import org.xtext.textualSlco.PortReference;
import org.xtext.textualSlco.PrimitiveType;
import org.xtext.textualSlco.SendSignal;
import org.xtext.textualSlco.SignalArgument;
import org.xtext.textualSlco.SignalArgumentExpression;
import org.xtext.textualSlco.SignalArgumentVariable;
import org.xtext.textualSlco.SignalReception;
import org.xtext.textualSlco.State;
import org.xtext.textualSlco.StateMachine;
import org.xtext.textualSlco.Statement;
import org.xtext.textualSlco.StringConstantExpression;
import org.xtext.textualSlco.TextualSlcoFactory;
import org.xtext.textualSlco.TextualSlcoPackage;
import org.xtext.textualSlco.Transition;
import org.xtext.textualSlco.UnidirectionalChannel;
import org.xtext.textualSlco.Variable;
import org.xtext.textualSlco.VariableExpression;
import org.xtext.textualSlco.VariableReference;
import org.xtext.textualSlco.Vertex;
import org.xtext.textualSlco.VertexReference;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class TextualSlcoPackageImpl extends EPackageImpl implements TextualSlcoPackage
{
  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass modelEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass classEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass stateMachineEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass vertexEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass initialEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass stateEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass finalEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass vertexReferenceEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass transitionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass statementEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass delayEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass portReferenceEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass signalReceptionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass signalArgumentEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass signalArgumentVariableEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass signalArgumentExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass assignmentEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass sendSignalEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass objectEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass portEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass channelEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass unidirectionalChannelEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass bidirectionalChannelEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass argumentTypeEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass variableEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass expressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass booleanConstantExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass integerConstantExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass stringConstantExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass variableReferenceEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass variableExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EClass binaryOperatorExpressionEClass = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EEnum primitiveTypeEEnum = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EEnum channelTypeEEnum = null;

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private EEnum operatorEEnum = null;

  /**
   * Creates an instance of the model <b>Package</b>, registered with
   * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
   * package URI value.
   * <p>Note: the correct way to create the package is via the static
   * factory method {@link #init init()}, which also performs
   * initialization of the package, or returns the registered package,
   * if one already exists.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.eclipse.emf.ecore.EPackage.Registry
   * @see org.xtext.textualSlco.TextualSlcoPackage#eNS_URI
   * @see #init()
   * @generated
   */
  private TextualSlcoPackageImpl()
  {
    super(eNS_URI, TextualSlcoFactory.eINSTANCE);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private static boolean isInited = false;

  /**
   * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
   * 
   * <p>This method is used to initialize {@link TextualSlcoPackage#eINSTANCE} when that field is accessed.
   * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see #eNS_URI
   * @see #createPackageContents()
   * @see #initializePackageContents()
   * @generated
   */
  public static TextualSlcoPackage init()
  {
    if (isInited) return (TextualSlcoPackage)EPackage.Registry.INSTANCE.getEPackage(TextualSlcoPackage.eNS_URI);

    // Obtain or create and register package
    TextualSlcoPackageImpl theTextualSlcoPackage = (TextualSlcoPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof TextualSlcoPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new TextualSlcoPackageImpl());

    isInited = true;

    // Create package meta-data objects
    theTextualSlcoPackage.createPackageContents();

    // Initialize created meta-data
    theTextualSlcoPackage.initializePackageContents();

    // Mark meta-data to indicate it can't be changed
    theTextualSlcoPackage.freeze();

  
    // Update the registry and return the package
    EPackage.Registry.INSTANCE.put(TextualSlcoPackage.eNS_URI, theTextualSlcoPackage);
    return theTextualSlcoPackage;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getModel()
  {
    return modelEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getModel_Name()
  {
    return (EAttribute)modelEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getModel_Classes()
  {
    return (EReference)modelEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getModel_Objects()
  {
    return (EReference)modelEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getModel_Channels()
  {
    return (EReference)modelEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getClass_()
  {
    return classEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getClass_Name()
  {
    return (EAttribute)classEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getClass_Variables()
  {
    return (EReference)classEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getClass_Ports()
  {
    return (EReference)classEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getClass_StateMachines()
  {
    return (EReference)classEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getStateMachine()
  {
    return stateMachineEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getStateMachine_Name()
  {
    return (EAttribute)stateMachineEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getStateMachine_Variables()
  {
    return (EReference)stateMachineEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getStateMachine_Vertices()
  {
    return (EReference)stateMachineEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getStateMachine_Transitions()
  {
    return (EReference)stateMachineEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getVertex()
  {
    return vertexEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getVertex_Name()
  {
    return (EAttribute)vertexEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getInitial()
  {
    return initialEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getState()
  {
    return stateEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getFinal()
  {
    return finalEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getVertexReference()
  {
    return vertexReferenceEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getVertexReference_Name()
  {
    return (EAttribute)vertexReferenceEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getTransition()
  {
    return transitionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getTransition_Name()
  {
    return (EAttribute)transitionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getTransition_Source()
  {
    return (EReference)transitionEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getTransition_Target()
  {
    return (EReference)transitionEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getTransition_Statements()
  {
    return (EReference)transitionEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getStatement()
  {
    return statementEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getDelay()
  {
    return delayEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getDelay_Value()
  {
    return (EAttribute)delayEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getPortReference()
  {
    return portReferenceEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getPortReference_Name()
  {
    return (EAttribute)portReferenceEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getSignalReception()
  {
    return signalReceptionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getSignalReception_SignalName()
  {
    return (EAttribute)signalReceptionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSignalReception_Arguments()
  {
    return (EReference)signalReceptionEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSignalReception_Condition()
  {
    return (EReference)signalReceptionEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSignalReception_Port()
  {
    return (EReference)signalReceptionEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getSignalArgument()
  {
    return signalArgumentEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getSignalArgumentVariable()
  {
    return signalArgumentVariableEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSignalArgumentVariable_Variable()
  {
    return (EReference)signalArgumentVariableEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getSignalArgumentExpression()
  {
    return signalArgumentExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSignalArgumentExpression_Expression()
  {
    return (EReference)signalArgumentExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getAssignment()
  {
    return assignmentEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getAssignment_Variable()
  {
    return (EReference)assignmentEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getAssignment_Expression()
  {
    return (EReference)assignmentEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getSendSignal()
  {
    return sendSignalEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getSendSignal_SignalName()
  {
    return (EAttribute)sendSignalEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSendSignal_Arguments()
  {
    return (EReference)sendSignalEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getSendSignal_Port()
  {
    return (EReference)sendSignalEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getObject()
  {
    return objectEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getObject_Name()
  {
    return (EAttribute)objectEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getObject_Class()
  {
    return (EReference)objectEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getPort()
  {
    return portEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getPort_Name()
  {
    return (EAttribute)portEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getChannel()
  {
    return channelEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getChannel_Name()
  {
    return (EAttribute)channelEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getChannel_ArgumentTypes()
  {
    return (EReference)channelEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getChannel_ChannelType()
  {
    return (EAttribute)channelEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getUnidirectionalChannel()
  {
    return unidirectionalChannelEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getUnidirectionalChannel_SourceObject()
  {
    return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getUnidirectionalChannel_SourcePort()
  {
    return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getUnidirectionalChannel_TargetObject()
  {
    return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getUnidirectionalChannel_TargetPort()
  {
    return (EReference)unidirectionalChannelEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getBidirectionalChannel()
  {
    return bidirectionalChannelEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBidirectionalChannel_Object1()
  {
    return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBidirectionalChannel_Port1()
  {
    return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBidirectionalChannel_Object2()
  {
    return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBidirectionalChannel_Port2()
  {
    return (EReference)bidirectionalChannelEClass.getEStructuralFeatures().get(3);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getArgumentType()
  {
    return argumentTypeEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getArgumentType_Type()
  {
    return (EAttribute)argumentTypeEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getVariable()
  {
    return variableEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getVariable_Type()
  {
    return (EAttribute)variableEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getVariable_Name()
  {
    return (EAttribute)variableEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getVariable_IntialValue()
  {
    return (EReference)variableEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getExpression()
  {
    return expressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getBooleanConstantExpression()
  {
    return booleanConstantExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getBooleanConstantExpression_Value()
  {
    return (EAttribute)booleanConstantExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getIntegerConstantExpression()
  {
    return integerConstantExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getIntegerConstantExpression_Value()
  {
    return (EAttribute)integerConstantExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getStringConstantExpression()
  {
    return stringConstantExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getStringConstantExpression_Value()
  {
    return (EAttribute)stringConstantExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getVariableReference()
  {
    return variableReferenceEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getVariableReference_Name()
  {
    return (EAttribute)variableReferenceEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getVariableExpression()
  {
    return variableExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getVariableExpression_Variable()
  {
    return (EReference)variableExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EClass getBinaryOperatorExpression()
  {
    return binaryOperatorExpressionEClass;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBinaryOperatorExpression_Operand1()
  {
    return (EReference)binaryOperatorExpressionEClass.getEStructuralFeatures().get(0);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EAttribute getBinaryOperatorExpression_Operator()
  {
    return (EAttribute)binaryOperatorExpressionEClass.getEStructuralFeatures().get(1);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EReference getBinaryOperatorExpression_Operand2()
  {
    return (EReference)binaryOperatorExpressionEClass.getEStructuralFeatures().get(2);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EEnum getPrimitiveType()
  {
    return primitiveTypeEEnum;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EEnum getChannelType()
  {
    return channelTypeEEnum;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public EEnum getOperator()
  {
    return operatorEEnum;
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public TextualSlcoFactory getTextualSlcoFactory()
  {
    return (TextualSlcoFactory)getEFactoryInstance();
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private boolean isCreated = false;

  /**
   * Creates the meta-model objects for the package.  This method is
   * guarded to have no affect on any invocation but its first.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void createPackageContents()
  {
    if (isCreated) return;
    isCreated = true;

    // Create classes and their features
    modelEClass = createEClass(MODEL);
    createEAttribute(modelEClass, MODEL__NAME);
    createEReference(modelEClass, MODEL__CLASSES);
    createEReference(modelEClass, MODEL__OBJECTS);
    createEReference(modelEClass, MODEL__CHANNELS);

    classEClass = createEClass(CLASS);
    createEAttribute(classEClass, CLASS__NAME);
    createEReference(classEClass, CLASS__VARIABLES);
    createEReference(classEClass, CLASS__PORTS);
    createEReference(classEClass, CLASS__STATE_MACHINES);

    stateMachineEClass = createEClass(STATE_MACHINE);
    createEAttribute(stateMachineEClass, STATE_MACHINE__NAME);
    createEReference(stateMachineEClass, STATE_MACHINE__VARIABLES);
    createEReference(stateMachineEClass, STATE_MACHINE__VERTICES);
    createEReference(stateMachineEClass, STATE_MACHINE__TRANSITIONS);

    vertexEClass = createEClass(VERTEX);
    createEAttribute(vertexEClass, VERTEX__NAME);

    initialEClass = createEClass(INITIAL);

    stateEClass = createEClass(STATE);

    finalEClass = createEClass(FINAL);

    vertexReferenceEClass = createEClass(VERTEX_REFERENCE);
    createEAttribute(vertexReferenceEClass, VERTEX_REFERENCE__NAME);

    transitionEClass = createEClass(TRANSITION);
    createEAttribute(transitionEClass, TRANSITION__NAME);
    createEReference(transitionEClass, TRANSITION__SOURCE);
    createEReference(transitionEClass, TRANSITION__TARGET);
    createEReference(transitionEClass, TRANSITION__STATEMENTS);

    statementEClass = createEClass(STATEMENT);

    delayEClass = createEClass(DELAY);
    createEAttribute(delayEClass, DELAY__VALUE);

    portReferenceEClass = createEClass(PORT_REFERENCE);
    createEAttribute(portReferenceEClass, PORT_REFERENCE__NAME);

    signalReceptionEClass = createEClass(SIGNAL_RECEPTION);
    createEAttribute(signalReceptionEClass, SIGNAL_RECEPTION__SIGNAL_NAME);
    createEReference(signalReceptionEClass, SIGNAL_RECEPTION__ARGUMENTS);
    createEReference(signalReceptionEClass, SIGNAL_RECEPTION__CONDITION);
    createEReference(signalReceptionEClass, SIGNAL_RECEPTION__PORT);

    signalArgumentEClass = createEClass(SIGNAL_ARGUMENT);

    signalArgumentVariableEClass = createEClass(SIGNAL_ARGUMENT_VARIABLE);
    createEReference(signalArgumentVariableEClass, SIGNAL_ARGUMENT_VARIABLE__VARIABLE);

    signalArgumentExpressionEClass = createEClass(SIGNAL_ARGUMENT_EXPRESSION);
    createEReference(signalArgumentExpressionEClass, SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION);

    assignmentEClass = createEClass(ASSIGNMENT);
    createEReference(assignmentEClass, ASSIGNMENT__VARIABLE);
    createEReference(assignmentEClass, ASSIGNMENT__EXPRESSION);

    sendSignalEClass = createEClass(SEND_SIGNAL);
    createEAttribute(sendSignalEClass, SEND_SIGNAL__SIGNAL_NAME);
    createEReference(sendSignalEClass, SEND_SIGNAL__ARGUMENTS);
    createEReference(sendSignalEClass, SEND_SIGNAL__PORT);

    objectEClass = createEClass(OBJECT);
    createEAttribute(objectEClass, OBJECT__NAME);
    createEReference(objectEClass, OBJECT__CLASS);

    portEClass = createEClass(PORT);
    createEAttribute(portEClass, PORT__NAME);

    channelEClass = createEClass(CHANNEL);
    createEAttribute(channelEClass, CHANNEL__NAME);
    createEReference(channelEClass, CHANNEL__ARGUMENT_TYPES);
    createEAttribute(channelEClass, CHANNEL__CHANNEL_TYPE);

    unidirectionalChannelEClass = createEClass(UNIDIRECTIONAL_CHANNEL);
    createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT);
    createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__SOURCE_PORT);
    createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT);
    createEReference(unidirectionalChannelEClass, UNIDIRECTIONAL_CHANNEL__TARGET_PORT);

    bidirectionalChannelEClass = createEClass(BIDIRECTIONAL_CHANNEL);
    createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__OBJECT1);
    createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__PORT1);
    createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__OBJECT2);
    createEReference(bidirectionalChannelEClass, BIDIRECTIONAL_CHANNEL__PORT2);

    argumentTypeEClass = createEClass(ARGUMENT_TYPE);
    createEAttribute(argumentTypeEClass, ARGUMENT_TYPE__TYPE);

    variableEClass = createEClass(VARIABLE);
    createEAttribute(variableEClass, VARIABLE__TYPE);
    createEAttribute(variableEClass, VARIABLE__NAME);
    createEReference(variableEClass, VARIABLE__INTIAL_VALUE);

    expressionEClass = createEClass(EXPRESSION);

    booleanConstantExpressionEClass = createEClass(BOOLEAN_CONSTANT_EXPRESSION);
    createEAttribute(booleanConstantExpressionEClass, BOOLEAN_CONSTANT_EXPRESSION__VALUE);

    integerConstantExpressionEClass = createEClass(INTEGER_CONSTANT_EXPRESSION);
    createEAttribute(integerConstantExpressionEClass, INTEGER_CONSTANT_EXPRESSION__VALUE);

    stringConstantExpressionEClass = createEClass(STRING_CONSTANT_EXPRESSION);
    createEAttribute(stringConstantExpressionEClass, STRING_CONSTANT_EXPRESSION__VALUE);

    variableReferenceEClass = createEClass(VARIABLE_REFERENCE);
    createEAttribute(variableReferenceEClass, VARIABLE_REFERENCE__NAME);

    variableExpressionEClass = createEClass(VARIABLE_EXPRESSION);
    createEReference(variableExpressionEClass, VARIABLE_EXPRESSION__VARIABLE);

    binaryOperatorExpressionEClass = createEClass(BINARY_OPERATOR_EXPRESSION);
    createEReference(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERAND1);
    createEAttribute(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERATOR);
    createEReference(binaryOperatorExpressionEClass, BINARY_OPERATOR_EXPRESSION__OPERAND2);

    // Create enums
    primitiveTypeEEnum = createEEnum(PRIMITIVE_TYPE);
    channelTypeEEnum = createEEnum(CHANNEL_TYPE);
    operatorEEnum = createEEnum(OPERATOR);
  }

  /**
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  private boolean isInitialized = false;

  /**
   * Complete the initialization of the package and its meta-model.  This
   * method is guarded to have no affect on any invocation but its first.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  public void initializePackageContents()
  {
    if (isInitialized) return;
    isInitialized = true;

    // Initialize package
    setName(eNAME);
    setNsPrefix(eNS_PREFIX);
    setNsURI(eNS_URI);

    // Create type parameters

    // Set bounds for type parameters

    // Add supertypes to classes
    initialEClass.getESuperTypes().add(this.getVertex());
    stateEClass.getESuperTypes().add(this.getVertex());
    finalEClass.getESuperTypes().add(this.getVertex());
    delayEClass.getESuperTypes().add(this.getStatement());
    signalReceptionEClass.getESuperTypes().add(this.getStatement());
    signalArgumentVariableEClass.getESuperTypes().add(this.getSignalArgument());
    signalArgumentExpressionEClass.getESuperTypes().add(this.getSignalArgument());
    assignmentEClass.getESuperTypes().add(this.getStatement());
    sendSignalEClass.getESuperTypes().add(this.getStatement());
    unidirectionalChannelEClass.getESuperTypes().add(this.getChannel());
    bidirectionalChannelEClass.getESuperTypes().add(this.getChannel());
    expressionEClass.getESuperTypes().add(this.getStatement());
    booleanConstantExpressionEClass.getESuperTypes().add(this.getExpression());
    integerConstantExpressionEClass.getESuperTypes().add(this.getExpression());
    stringConstantExpressionEClass.getESuperTypes().add(this.getExpression());
    variableExpressionEClass.getESuperTypes().add(this.getExpression());
    binaryOperatorExpressionEClass.getESuperTypes().add(this.getExpression());

    // Initialize classes and features; add operations and parameters
    initEClass(modelEClass, Model.class, "Model", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getModel_Name(), ecorePackage.getEString(), "name", null, 0, 1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getModel_Classes(), this.getClass_(), null, "classes", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getModel_Objects(), this.getObject(), null, "objects", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getModel_Channels(), this.getChannel(), null, "channels", null, 0, -1, Model.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(classEClass, org.xtext.textualSlco.Class.class, "Class", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getClass_Name(), ecorePackage.getEString(), "name", null, 0, 1, org.xtext.textualSlco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getClass_Variables(), this.getVariable(), null, "variables", null, 0, -1, org.xtext.textualSlco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getClass_Ports(), this.getPort(), null, "ports", null, 0, -1, org.xtext.textualSlco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getClass_StateMachines(), this.getStateMachine(), null, "stateMachines", null, 0, -1, org.xtext.textualSlco.Class.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(stateMachineEClass, StateMachine.class, "StateMachine", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getStateMachine_Name(), ecorePackage.getEString(), "name", null, 0, 1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getStateMachine_Variables(), this.getVariable(), null, "variables", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getStateMachine_Vertices(), this.getVertex(), null, "vertices", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getStateMachine_Transitions(), this.getTransition(), null, "transitions", null, 0, -1, StateMachine.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(vertexEClass, Vertex.class, "Vertex", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getVertex_Name(), ecorePackage.getEString(), "name", null, 0, 1, Vertex.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(initialEClass, Initial.class, "Initial", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(stateEClass, State.class, "State", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(finalEClass, Final.class, "Final", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(vertexReferenceEClass, VertexReference.class, "VertexReference", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getVertexReference_Name(), ecorePackage.getEString(), "name", null, 0, 1, VertexReference.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(transitionEClass, Transition.class, "Transition", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getTransition_Name(), ecorePackage.getEString(), "name", null, 0, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getTransition_Source(), this.getVertexReference(), null, "source", null, 0, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getTransition_Target(), this.getVertexReference(), null, "target", null, 0, 1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getTransition_Statements(), this.getStatement(), null, "statements", null, 0, -1, Transition.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(statementEClass, Statement.class, "Statement", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(delayEClass, Delay.class, "Delay", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getDelay_Value(), ecorePackage.getEInt(), "value", null, 0, 1, Delay.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(portReferenceEClass, PortReference.class, "PortReference", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getPortReference_Name(), ecorePackage.getEString(), "name", null, 0, 1, PortReference.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(signalReceptionEClass, SignalReception.class, "SignalReception", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getSignalReception_SignalName(), ecorePackage.getEString(), "signalName", null, 0, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getSignalReception_Arguments(), this.getSignalArgument(), null, "arguments", null, 0, -1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getSignalReception_Condition(), this.getExpression(), null, "condition", null, 0, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getSignalReception_Port(), this.getPortReference(), null, "port", null, 0, 1, SignalReception.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(signalArgumentEClass, SignalArgument.class, "SignalArgument", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(signalArgumentVariableEClass, SignalArgumentVariable.class, "SignalArgumentVariable", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getSignalArgumentVariable_Variable(), this.getVariableReference(), null, "variable", null, 0, 1, SignalArgumentVariable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(signalArgumentExpressionEClass, SignalArgumentExpression.class, "SignalArgumentExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getSignalArgumentExpression_Expression(), this.getExpression(), null, "expression", null, 0, 1, SignalArgumentExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(assignmentEClass, Assignment.class, "Assignment", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getAssignment_Variable(), this.getVariableReference(), null, "variable", null, 0, 1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getAssignment_Expression(), this.getExpression(), null, "expression", null, 0, 1, Assignment.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(sendSignalEClass, SendSignal.class, "SendSignal", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getSendSignal_SignalName(), ecorePackage.getEString(), "signalName", null, 0, 1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getSendSignal_Arguments(), this.getExpression(), null, "arguments", null, 0, -1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getSendSignal_Port(), this.getPortReference(), null, "port", null, 0, 1, SendSignal.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(objectEClass, org.xtext.textualSlco.Object.class, "Object", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getObject_Name(), ecorePackage.getEString(), "name", null, 0, 1, org.xtext.textualSlco.Object.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getObject_Class(), this.getClass_(), null, "class", null, 0, 1, org.xtext.textualSlco.Object.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(portEClass, Port.class, "Port", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getPort_Name(), ecorePackage.getEString(), "name", null, 0, 1, Port.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(channelEClass, Channel.class, "Channel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getChannel_Name(), ecorePackage.getEString(), "name", null, 0, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getChannel_ArgumentTypes(), this.getArgumentType(), null, "argumentTypes", null, 0, -1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEAttribute(getChannel_ChannelType(), this.getChannelType(), "channelType", null, 0, 1, Channel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(unidirectionalChannelEClass, UnidirectionalChannel.class, "UnidirectionalChannel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getUnidirectionalChannel_SourceObject(), this.getObject(), null, "sourceObject", null, 0, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getUnidirectionalChannel_SourcePort(), this.getPortReference(), null, "sourcePort", null, 0, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getUnidirectionalChannel_TargetObject(), this.getObject(), null, "targetObject", null, 0, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getUnidirectionalChannel_TargetPort(), this.getPortReference(), null, "targetPort", null, 0, 1, UnidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(bidirectionalChannelEClass, BidirectionalChannel.class, "BidirectionalChannel", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getBidirectionalChannel_Object1(), this.getObject(), null, "object1", null, 0, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getBidirectionalChannel_Port1(), this.getPortReference(), null, "port1", null, 0, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getBidirectionalChannel_Object2(), this.getObject(), null, "object2", null, 0, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getBidirectionalChannel_Port2(), this.getPortReference(), null, "port2", null, 0, 1, BidirectionalChannel.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(argumentTypeEClass, ArgumentType.class, "ArgumentType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getArgumentType_Type(), this.getPrimitiveType(), "type", null, 0, 1, ArgumentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(variableEClass, Variable.class, "Variable", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getVariable_Type(), this.getPrimitiveType(), "type", null, 0, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEAttribute(getVariable_Name(), ecorePackage.getEString(), "name", null, 0, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getVariable_IntialValue(), this.getExpression(), null, "intialValue", null, 0, 1, Variable.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(expressionEClass, Expression.class, "Expression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

    initEClass(booleanConstantExpressionEClass, BooleanConstantExpression.class, "BooleanConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getBooleanConstantExpression_Value(), ecorePackage.getEString(), "value", null, 0, 1, BooleanConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(integerConstantExpressionEClass, IntegerConstantExpression.class, "IntegerConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getIntegerConstantExpression_Value(), ecorePackage.getEInt(), "value", null, 0, 1, IntegerConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(stringConstantExpressionEClass, StringConstantExpression.class, "StringConstantExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getStringConstantExpression_Value(), ecorePackage.getEString(), "value", null, 0, 1, StringConstantExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(variableReferenceEClass, VariableReference.class, "VariableReference", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEAttribute(getVariableReference_Name(), ecorePackage.getEString(), "name", null, 0, 1, VariableReference.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(variableExpressionEClass, VariableExpression.class, "VariableExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getVariableExpression_Variable(), this.getVariableReference(), null, "variable", null, 0, 1, VariableExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    initEClass(binaryOperatorExpressionEClass, BinaryOperatorExpression.class, "BinaryOperatorExpression", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
    initEReference(getBinaryOperatorExpression_Operand1(), this.getExpression(), null, "operand1", null, 0, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEAttribute(getBinaryOperatorExpression_Operator(), this.getOperator(), "operator", null, 0, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
    initEReference(getBinaryOperatorExpression_Operand2(), this.getExpression(), null, "operand2", null, 0, 1, BinaryOperatorExpression.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

    // Initialize enums and add enum literals
    initEEnum(primitiveTypeEEnum, PrimitiveType.class, "PrimitiveType");
    addEEnumLiteral(primitiveTypeEEnum, PrimitiveType.INTEGER);
    addEEnumLiteral(primitiveTypeEEnum, PrimitiveType.STRING);
    addEEnumLiteral(primitiveTypeEEnum, PrimitiveType.BOOLEAN);

    initEEnum(channelTypeEEnum, ChannelType.class, "ChannelType");
    addEEnumLiteral(channelTypeEEnum, ChannelType.SYNCHRONOUS);
    addEEnumLiteral(channelTypeEEnum, ChannelType.ASYNCHRONOUS_LOSSLESS);
    addEEnumLiteral(channelTypeEEnum, ChannelType.ASYNCHRONOUS_LOSSY);

    initEEnum(operatorEEnum, Operator.class, "Operator");
    addEEnumLiteral(operatorEEnum, Operator.AT_LEAST);
    addEEnumLiteral(operatorEEnum, Operator.AT_MOST);
    addEEnumLiteral(operatorEEnum, Operator.ADD);
    addEEnumLiteral(operatorEEnum, Operator.AND);
    addEEnumLiteral(operatorEEnum, Operator.OR);
    addEEnumLiteral(operatorEEnum, Operator.EQUALS);
    addEEnumLiteral(operatorEEnum, Operator.DIFFERS);
    addEEnumLiteral(operatorEEnum, Operator.SUBTRACT);

    // Create resource
    createResource(eNS_URI);
  }

} //TextualSlcoPackageImpl

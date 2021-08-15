/**
 */
package org.xtext.textualSlco;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see org.xtext.textualSlco.TextualSlcoFactory
 * @model kind="package"
 * @generated
 */
public interface TextualSlcoPackage extends EPackage
{
  /**
   * The package name.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  String eNAME = "textualSlco";

  /**
   * The package namespace URI.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  String eNS_URI = "http://www.xtext.org/TextualSlco";

  /**
   * The package namespace name.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  String eNS_PREFIX = "textualSlco";

  /**
   * The singleton instance of the package.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   */
  TextualSlcoPackage eINSTANCE = org.xtext.textualSlco.impl.TextualSlcoPackageImpl.init();

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ModelImpl <em>Model</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ModelImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getModel()
   * @generated
   */
  int MODEL = 0;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int MODEL__NAME = 0;

  /**
   * The feature id for the '<em><b>Classes</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int MODEL__CLASSES = 1;

  /**
   * The feature id for the '<em><b>Objects</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int MODEL__OBJECTS = 2;

  /**
   * The feature id for the '<em><b>Channels</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int MODEL__CHANNELS = 3;

  /**
   * The number of structural features of the '<em>Model</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int MODEL_FEATURE_COUNT = 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ClassImpl <em>Class</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ClassImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getClass_()
   * @generated
   */
  int CLASS = 1;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CLASS__NAME = 0;

  /**
   * The feature id for the '<em><b>Variables</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CLASS__VARIABLES = 1;

  /**
   * The feature id for the '<em><b>Ports</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CLASS__PORTS = 2;

  /**
   * The feature id for the '<em><b>State Machines</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CLASS__STATE_MACHINES = 3;

  /**
   * The number of structural features of the '<em>Class</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CLASS_FEATURE_COUNT = 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.StateMachineImpl <em>State Machine</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.StateMachineImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStateMachine()
   * @generated
   */
  int STATE_MACHINE = 2;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_MACHINE__NAME = 0;

  /**
   * The feature id for the '<em><b>Variables</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_MACHINE__VARIABLES = 1;

  /**
   * The feature id for the '<em><b>Vertices</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_MACHINE__VERTICES = 2;

  /**
   * The feature id for the '<em><b>Transitions</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_MACHINE__TRANSITIONS = 3;

  /**
   * The number of structural features of the '<em>State Machine</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_MACHINE_FEATURE_COUNT = 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.VertexImpl <em>Vertex</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.VertexImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVertex()
   * @generated
   */
  int VERTEX = 3;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VERTEX__NAME = 0;

  /**
   * The number of structural features of the '<em>Vertex</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VERTEX_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.InitialImpl <em>Initial</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.InitialImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getInitial()
   * @generated
   */
  int INITIAL = 4;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int INITIAL__NAME = VERTEX__NAME;

  /**
   * The number of structural features of the '<em>Initial</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int INITIAL_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.StateImpl <em>State</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.StateImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getState()
   * @generated
   */
  int STATE = 5;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE__NAME = VERTEX__NAME;

  /**
   * The number of structural features of the '<em>State</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATE_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.FinalImpl <em>Final</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.FinalImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getFinal()
   * @generated
   */
  int FINAL = 6;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int FINAL__NAME = VERTEX__NAME;

  /**
   * The number of structural features of the '<em>Final</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int FINAL_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.VertexReferenceImpl <em>Vertex Reference</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.VertexReferenceImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVertexReference()
   * @generated
   */
  int VERTEX_REFERENCE = 7;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VERTEX_REFERENCE__NAME = 0;

  /**
   * The number of structural features of the '<em>Vertex Reference</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VERTEX_REFERENCE_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.TransitionImpl <em>Transition</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.TransitionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getTransition()
   * @generated
   */
  int TRANSITION = 8;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int TRANSITION__NAME = 0;

  /**
   * The feature id for the '<em><b>Source</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int TRANSITION__SOURCE = 1;

  /**
   * The feature id for the '<em><b>Target</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int TRANSITION__TARGET = 2;

  /**
   * The feature id for the '<em><b>Statements</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int TRANSITION__STATEMENTS = 3;

  /**
   * The number of structural features of the '<em>Transition</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int TRANSITION_FEATURE_COUNT = 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.StatementImpl <em>Statement</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.StatementImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStatement()
   * @generated
   */
  int STATEMENT = 9;

  /**
   * The number of structural features of the '<em>Statement</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STATEMENT_FEATURE_COUNT = 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.DelayImpl <em>Delay</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.DelayImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getDelay()
   * @generated
   */
  int DELAY = 10;

  /**
   * The feature id for the '<em><b>Value</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int DELAY__VALUE = STATEMENT_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Delay</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int DELAY_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.PortReferenceImpl <em>Port Reference</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.PortReferenceImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPortReference()
   * @generated
   */
  int PORT_REFERENCE = 11;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int PORT_REFERENCE__NAME = 0;

  /**
   * The number of structural features of the '<em>Port Reference</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int PORT_REFERENCE_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.SignalReceptionImpl <em>Signal Reception</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.SignalReceptionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalReception()
   * @generated
   */
  int SIGNAL_RECEPTION = 12;

  /**
   * The feature id for the '<em><b>Signal Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_RECEPTION__SIGNAL_NAME = STATEMENT_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Arguments</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_RECEPTION__ARGUMENTS = STATEMENT_FEATURE_COUNT + 1;

  /**
   * The feature id for the '<em><b>Condition</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_RECEPTION__CONDITION = STATEMENT_FEATURE_COUNT + 2;

  /**
   * The feature id for the '<em><b>Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_RECEPTION__PORT = STATEMENT_FEATURE_COUNT + 3;

  /**
   * The number of structural features of the '<em>Signal Reception</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_RECEPTION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.SignalArgumentImpl <em>Signal Argument</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.SignalArgumentImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgument()
   * @generated
   */
  int SIGNAL_ARGUMENT = 13;

  /**
   * The number of structural features of the '<em>Signal Argument</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_ARGUMENT_FEATURE_COUNT = 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.SignalArgumentVariableImpl <em>Signal Argument Variable</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.SignalArgumentVariableImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgumentVariable()
   * @generated
   */
  int SIGNAL_ARGUMENT_VARIABLE = 14;

  /**
   * The feature id for the '<em><b>Variable</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_ARGUMENT_VARIABLE__VARIABLE = SIGNAL_ARGUMENT_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Signal Argument Variable</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_ARGUMENT_VARIABLE_FEATURE_COUNT = SIGNAL_ARGUMENT_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.SignalArgumentExpressionImpl <em>Signal Argument Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.SignalArgumentExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgumentExpression()
   * @generated
   */
  int SIGNAL_ARGUMENT_EXPRESSION = 15;

  /**
   * The feature id for the '<em><b>Expression</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION = SIGNAL_ARGUMENT_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Signal Argument Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SIGNAL_ARGUMENT_EXPRESSION_FEATURE_COUNT = SIGNAL_ARGUMENT_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.AssignmentImpl <em>Assignment</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.AssignmentImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getAssignment()
   * @generated
   */
  int ASSIGNMENT = 16;

  /**
   * The feature id for the '<em><b>Variable</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int ASSIGNMENT__VARIABLE = STATEMENT_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Expression</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int ASSIGNMENT__EXPRESSION = STATEMENT_FEATURE_COUNT + 1;

  /**
   * The number of structural features of the '<em>Assignment</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int ASSIGNMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 2;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.SendSignalImpl <em>Send Signal</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.SendSignalImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSendSignal()
   * @generated
   */
  int SEND_SIGNAL = 17;

  /**
   * The feature id for the '<em><b>Signal Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SEND_SIGNAL__SIGNAL_NAME = STATEMENT_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Arguments</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SEND_SIGNAL__ARGUMENTS = STATEMENT_FEATURE_COUNT + 1;

  /**
   * The feature id for the '<em><b>Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SEND_SIGNAL__PORT = STATEMENT_FEATURE_COUNT + 2;

  /**
   * The number of structural features of the '<em>Send Signal</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int SEND_SIGNAL_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ObjectImpl <em>Object</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ObjectImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getObject()
   * @generated
   */
  int OBJECT = 18;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int OBJECT__NAME = 0;

  /**
   * The feature id for the '<em><b>Class</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int OBJECT__CLASS = 1;

  /**
   * The number of structural features of the '<em>Object</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int OBJECT_FEATURE_COUNT = 2;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.PortImpl <em>Port</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.PortImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPort()
   * @generated
   */
  int PORT = 19;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int PORT__NAME = 0;

  /**
   * The number of structural features of the '<em>Port</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int PORT_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ChannelImpl <em>Channel</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ChannelImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getChannel()
   * @generated
   */
  int CHANNEL = 20;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CHANNEL__NAME = 0;

  /**
   * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CHANNEL__ARGUMENT_TYPES = 1;

  /**
   * The feature id for the '<em><b>Channel Type</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CHANNEL__CHANNEL_TYPE = 2;

  /**
   * The number of structural features of the '<em>Channel</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int CHANNEL_FEATURE_COUNT = 3;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl <em>Unidirectional Channel</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.UnidirectionalChannelImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getUnidirectionalChannel()
   * @generated
   */
  int UNIDIRECTIONAL_CHANNEL = 21;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__NAME = CHANNEL__NAME;

  /**
   * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__ARGUMENT_TYPES = CHANNEL__ARGUMENT_TYPES;

  /**
   * The feature id for the '<em><b>Channel Type</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__CHANNEL_TYPE = CHANNEL__CHANNEL_TYPE;

  /**
   * The feature id for the '<em><b>Source Object</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT = CHANNEL_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Source Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__SOURCE_PORT = CHANNEL_FEATURE_COUNT + 1;

  /**
   * The feature id for the '<em><b>Target Object</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT = CHANNEL_FEATURE_COUNT + 2;

  /**
   * The feature id for the '<em><b>Target Port</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL__TARGET_PORT = CHANNEL_FEATURE_COUNT + 3;

  /**
   * The number of structural features of the '<em>Unidirectional Channel</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int UNIDIRECTIONAL_CHANNEL_FEATURE_COUNT = CHANNEL_FEATURE_COUNT + 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl <em>Bidirectional Channel</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.BidirectionalChannelImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBidirectionalChannel()
   * @generated
   */
  int BIDIRECTIONAL_CHANNEL = 22;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__NAME = CHANNEL__NAME;

  /**
   * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__ARGUMENT_TYPES = CHANNEL__ARGUMENT_TYPES;

  /**
   * The feature id for the '<em><b>Channel Type</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__CHANNEL_TYPE = CHANNEL__CHANNEL_TYPE;

  /**
   * The feature id for the '<em><b>Object1</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__OBJECT1 = CHANNEL_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Port1</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__PORT1 = CHANNEL_FEATURE_COUNT + 1;

  /**
   * The feature id for the '<em><b>Object2</b></em>' reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__OBJECT2 = CHANNEL_FEATURE_COUNT + 2;

  /**
   * The feature id for the '<em><b>Port2</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL__PORT2 = CHANNEL_FEATURE_COUNT + 3;

  /**
   * The number of structural features of the '<em>Bidirectional Channel</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BIDIRECTIONAL_CHANNEL_FEATURE_COUNT = CHANNEL_FEATURE_COUNT + 4;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ArgumentTypeImpl <em>Argument Type</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ArgumentTypeImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getArgumentType()
   * @generated
   */
  int ARGUMENT_TYPE = 23;

  /**
   * The feature id for the '<em><b>Type</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int ARGUMENT_TYPE__TYPE = 0;

  /**
   * The number of structural features of the '<em>Argument Type</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int ARGUMENT_TYPE_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.VariableImpl <em>Variable</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.VariableImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariable()
   * @generated
   */
  int VARIABLE = 24;

  /**
   * The feature id for the '<em><b>Type</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE__TYPE = 0;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE__NAME = 1;

  /**
   * The feature id for the '<em><b>Intial Value</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE__INTIAL_VALUE = 2;

  /**
   * The number of structural features of the '<em>Variable</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE_FEATURE_COUNT = 3;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.ExpressionImpl <em>Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.ExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getExpression()
   * @generated
   */
  int EXPRESSION = 25;

  /**
   * The number of structural features of the '<em>Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int EXPRESSION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.BooleanConstantExpressionImpl <em>Boolean Constant Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.BooleanConstantExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBooleanConstantExpression()
   * @generated
   */
  int BOOLEAN_CONSTANT_EXPRESSION = 26;

  /**
   * The feature id for the '<em><b>Value</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BOOLEAN_CONSTANT_EXPRESSION__VALUE = EXPRESSION_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Boolean Constant Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BOOLEAN_CONSTANT_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.IntegerConstantExpressionImpl <em>Integer Constant Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.IntegerConstantExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getIntegerConstantExpression()
   * @generated
   */
  int INTEGER_CONSTANT_EXPRESSION = 27;

  /**
   * The feature id for the '<em><b>Value</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int INTEGER_CONSTANT_EXPRESSION__VALUE = EXPRESSION_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Integer Constant Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int INTEGER_CONSTANT_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.StringConstantExpressionImpl <em>String Constant Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.StringConstantExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStringConstantExpression()
   * @generated
   */
  int STRING_CONSTANT_EXPRESSION = 28;

  /**
   * The feature id for the '<em><b>Value</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STRING_CONSTANT_EXPRESSION__VALUE = EXPRESSION_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>String Constant Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int STRING_CONSTANT_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.VariableReferenceImpl <em>Variable Reference</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.VariableReferenceImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariableReference()
   * @generated
   */
  int VARIABLE_REFERENCE = 29;

  /**
   * The feature id for the '<em><b>Name</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE_REFERENCE__NAME = 0;

  /**
   * The number of structural features of the '<em>Variable Reference</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE_REFERENCE_FEATURE_COUNT = 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.VariableExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariableExpression()
   * @generated
   */
  int VARIABLE_EXPRESSION = 30;

  /**
   * The feature id for the '<em><b>Variable</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE_EXPRESSION__VARIABLE = EXPRESSION_FEATURE_COUNT + 0;

  /**
   * The number of structural features of the '<em>Variable Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int VARIABLE_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 1;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.impl.BinaryOperatorExpressionImpl <em>Binary Operator Expression</em>}' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.impl.BinaryOperatorExpressionImpl
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBinaryOperatorExpression()
   * @generated
   */
  int BINARY_OPERATOR_EXPRESSION = 31;

  /**
   * The feature id for the '<em><b>Operand1</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BINARY_OPERATOR_EXPRESSION__OPERAND1 = EXPRESSION_FEATURE_COUNT + 0;

  /**
   * The feature id for the '<em><b>Operator</b></em>' attribute.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BINARY_OPERATOR_EXPRESSION__OPERATOR = EXPRESSION_FEATURE_COUNT + 1;

  /**
   * The feature id for the '<em><b>Operand2</b></em>' containment reference.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BINARY_OPERATOR_EXPRESSION__OPERAND2 = EXPRESSION_FEATURE_COUNT + 2;

  /**
   * The number of structural features of the '<em>Binary Operator Expression</em>' class.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @generated
   * @ordered
   */
  int BINARY_OPERATOR_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 3;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.PrimitiveType <em>Primitive Type</em>}' enum.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.PrimitiveType
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPrimitiveType()
   * @generated
   */
  int PRIMITIVE_TYPE = 32;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.ChannelType <em>Channel Type</em>}' enum.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.ChannelType
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getChannelType()
   * @generated
   */
  int CHANNEL_TYPE = 33;

  /**
   * The meta object id for the '{@link org.xtext.textualSlco.Operator <em>Operator</em>}' enum.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @see org.xtext.textualSlco.Operator
   * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getOperator()
   * @generated
   */
  int OPERATOR = 34;


  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Model <em>Model</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Model</em>'.
   * @see org.xtext.textualSlco.Model
   * @generated
   */
  EClass getModel();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Model#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Model#getName()
   * @see #getModel()
   * @generated
   */
  EAttribute getModel_Name();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Model#getClasses <em>Classes</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Classes</em>'.
   * @see org.xtext.textualSlco.Model#getClasses()
   * @see #getModel()
   * @generated
   */
  EReference getModel_Classes();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Model#getObjects <em>Objects</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Objects</em>'.
   * @see org.xtext.textualSlco.Model#getObjects()
   * @see #getModel()
   * @generated
   */
  EReference getModel_Objects();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Model#getChannels <em>Channels</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Channels</em>'.
   * @see org.xtext.textualSlco.Model#getChannels()
   * @see #getModel()
   * @generated
   */
  EReference getModel_Channels();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Class <em>Class</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Class</em>'.
   * @see org.xtext.textualSlco.Class
   * @generated
   */
  EClass getClass_();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Class#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Class#getName()
   * @see #getClass_()
   * @generated
   */
  EAttribute getClass_Name();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Class#getVariables <em>Variables</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Variables</em>'.
   * @see org.xtext.textualSlco.Class#getVariables()
   * @see #getClass_()
   * @generated
   */
  EReference getClass_Variables();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Class#getPorts <em>Ports</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Ports</em>'.
   * @see org.xtext.textualSlco.Class#getPorts()
   * @see #getClass_()
   * @generated
   */
  EReference getClass_Ports();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Class#getStateMachines <em>State Machines</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>State Machines</em>'.
   * @see org.xtext.textualSlco.Class#getStateMachines()
   * @see #getClass_()
   * @generated
   */
  EReference getClass_StateMachines();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.StateMachine <em>State Machine</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>State Machine</em>'.
   * @see org.xtext.textualSlco.StateMachine
   * @generated
   */
  EClass getStateMachine();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.StateMachine#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.StateMachine#getName()
   * @see #getStateMachine()
   * @generated
   */
  EAttribute getStateMachine_Name();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.StateMachine#getVariables <em>Variables</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Variables</em>'.
   * @see org.xtext.textualSlco.StateMachine#getVariables()
   * @see #getStateMachine()
   * @generated
   */
  EReference getStateMachine_Variables();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.StateMachine#getVertices <em>Vertices</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Vertices</em>'.
   * @see org.xtext.textualSlco.StateMachine#getVertices()
   * @see #getStateMachine()
   * @generated
   */
  EReference getStateMachine_Vertices();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.StateMachine#getTransitions <em>Transitions</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Transitions</em>'.
   * @see org.xtext.textualSlco.StateMachine#getTransitions()
   * @see #getStateMachine()
   * @generated
   */
  EReference getStateMachine_Transitions();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Vertex <em>Vertex</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Vertex</em>'.
   * @see org.xtext.textualSlco.Vertex
   * @generated
   */
  EClass getVertex();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Vertex#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Vertex#getName()
   * @see #getVertex()
   * @generated
   */
  EAttribute getVertex_Name();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Initial <em>Initial</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Initial</em>'.
   * @see org.xtext.textualSlco.Initial
   * @generated
   */
  EClass getInitial();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.State <em>State</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>State</em>'.
   * @see org.xtext.textualSlco.State
   * @generated
   */
  EClass getState();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Final <em>Final</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Final</em>'.
   * @see org.xtext.textualSlco.Final
   * @generated
   */
  EClass getFinal();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.VertexReference <em>Vertex Reference</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Vertex Reference</em>'.
   * @see org.xtext.textualSlco.VertexReference
   * @generated
   */
  EClass getVertexReference();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.VertexReference#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.VertexReference#getName()
   * @see #getVertexReference()
   * @generated
   */
  EAttribute getVertexReference_Name();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Transition <em>Transition</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Transition</em>'.
   * @see org.xtext.textualSlco.Transition
   * @generated
   */
  EClass getTransition();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Transition#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Transition#getName()
   * @see #getTransition()
   * @generated
   */
  EAttribute getTransition_Name();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.Transition#getSource <em>Source</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Source</em>'.
   * @see org.xtext.textualSlco.Transition#getSource()
   * @see #getTransition()
   * @generated
   */
  EReference getTransition_Source();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.Transition#getTarget <em>Target</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Target</em>'.
   * @see org.xtext.textualSlco.Transition#getTarget()
   * @see #getTransition()
   * @generated
   */
  EReference getTransition_Target();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Transition#getStatements <em>Statements</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Statements</em>'.
   * @see org.xtext.textualSlco.Transition#getStatements()
   * @see #getTransition()
   * @generated
   */
  EReference getTransition_Statements();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Statement <em>Statement</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Statement</em>'.
   * @see org.xtext.textualSlco.Statement
   * @generated
   */
  EClass getStatement();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Delay <em>Delay</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Delay</em>'.
   * @see org.xtext.textualSlco.Delay
   * @generated
   */
  EClass getDelay();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Delay#getValue <em>Value</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Value</em>'.
   * @see org.xtext.textualSlco.Delay#getValue()
   * @see #getDelay()
   * @generated
   */
  EAttribute getDelay_Value();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.PortReference <em>Port Reference</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Port Reference</em>'.
   * @see org.xtext.textualSlco.PortReference
   * @generated
   */
  EClass getPortReference();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.PortReference#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.PortReference#getName()
   * @see #getPortReference()
   * @generated
   */
  EAttribute getPortReference_Name();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.SignalReception <em>Signal Reception</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Signal Reception</em>'.
   * @see org.xtext.textualSlco.SignalReception
   * @generated
   */
  EClass getSignalReception();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.SignalReception#getSignalName <em>Signal Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Signal Name</em>'.
   * @see org.xtext.textualSlco.SignalReception#getSignalName()
   * @see #getSignalReception()
   * @generated
   */
  EAttribute getSignalReception_SignalName();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.SignalReception#getArguments <em>Arguments</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Arguments</em>'.
   * @see org.xtext.textualSlco.SignalReception#getArguments()
   * @see #getSignalReception()
   * @generated
   */
  EReference getSignalReception_Arguments();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.SignalReception#getCondition <em>Condition</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Condition</em>'.
   * @see org.xtext.textualSlco.SignalReception#getCondition()
   * @see #getSignalReception()
   * @generated
   */
  EReference getSignalReception_Condition();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.SignalReception#getPort <em>Port</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Port</em>'.
   * @see org.xtext.textualSlco.SignalReception#getPort()
   * @see #getSignalReception()
   * @generated
   */
  EReference getSignalReception_Port();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.SignalArgument <em>Signal Argument</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Signal Argument</em>'.
   * @see org.xtext.textualSlco.SignalArgument
   * @generated
   */
  EClass getSignalArgument();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.SignalArgumentVariable <em>Signal Argument Variable</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Signal Argument Variable</em>'.
   * @see org.xtext.textualSlco.SignalArgumentVariable
   * @generated
   */
  EClass getSignalArgumentVariable();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.SignalArgumentVariable#getVariable <em>Variable</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Variable</em>'.
   * @see org.xtext.textualSlco.SignalArgumentVariable#getVariable()
   * @see #getSignalArgumentVariable()
   * @generated
   */
  EReference getSignalArgumentVariable_Variable();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.SignalArgumentExpression <em>Signal Argument Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Signal Argument Expression</em>'.
   * @see org.xtext.textualSlco.SignalArgumentExpression
   * @generated
   */
  EClass getSignalArgumentExpression();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.SignalArgumentExpression#getExpression <em>Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Expression</em>'.
   * @see org.xtext.textualSlco.SignalArgumentExpression#getExpression()
   * @see #getSignalArgumentExpression()
   * @generated
   */
  EReference getSignalArgumentExpression_Expression();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Assignment <em>Assignment</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Assignment</em>'.
   * @see org.xtext.textualSlco.Assignment
   * @generated
   */
  EClass getAssignment();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.Assignment#getVariable <em>Variable</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Variable</em>'.
   * @see org.xtext.textualSlco.Assignment#getVariable()
   * @see #getAssignment()
   * @generated
   */
  EReference getAssignment_Variable();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.Assignment#getExpression <em>Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Expression</em>'.
   * @see org.xtext.textualSlco.Assignment#getExpression()
   * @see #getAssignment()
   * @generated
   */
  EReference getAssignment_Expression();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.SendSignal <em>Send Signal</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Send Signal</em>'.
   * @see org.xtext.textualSlco.SendSignal
   * @generated
   */
  EClass getSendSignal();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.SendSignal#getSignalName <em>Signal Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Signal Name</em>'.
   * @see org.xtext.textualSlco.SendSignal#getSignalName()
   * @see #getSendSignal()
   * @generated
   */
  EAttribute getSendSignal_SignalName();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.SendSignal#getArguments <em>Arguments</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Arguments</em>'.
   * @see org.xtext.textualSlco.SendSignal#getArguments()
   * @see #getSendSignal()
   * @generated
   */
  EReference getSendSignal_Arguments();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.SendSignal#getPort <em>Port</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Port</em>'.
   * @see org.xtext.textualSlco.SendSignal#getPort()
   * @see #getSendSignal()
   * @generated
   */
  EReference getSendSignal_Port();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Object <em>Object</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Object</em>'.
   * @see org.xtext.textualSlco.Object
   * @generated
   */
  EClass getObject();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Object#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Object#getName()
   * @see #getObject()
   * @generated
   */
  EAttribute getObject_Name();

  /**
   * Returns the meta object for the reference '{@link org.xtext.textualSlco.Object#getClass_ <em>Class</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the reference '<em>Class</em>'.
   * @see org.xtext.textualSlco.Object#getClass_()
   * @see #getObject()
   * @generated
   */
  EReference getObject_Class();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Port <em>Port</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Port</em>'.
   * @see org.xtext.textualSlco.Port
   * @generated
   */
  EClass getPort();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Port#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Port#getName()
   * @see #getPort()
   * @generated
   */
  EAttribute getPort_Name();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Channel <em>Channel</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Channel</em>'.
   * @see org.xtext.textualSlco.Channel
   * @generated
   */
  EClass getChannel();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Channel#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Channel#getName()
   * @see #getChannel()
   * @generated
   */
  EAttribute getChannel_Name();

  /**
   * Returns the meta object for the containment reference list '{@link org.xtext.textualSlco.Channel#getArgumentTypes <em>Argument Types</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference list '<em>Argument Types</em>'.
   * @see org.xtext.textualSlco.Channel#getArgumentTypes()
   * @see #getChannel()
   * @generated
   */
  EReference getChannel_ArgumentTypes();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Channel#getChannelType <em>Channel Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Channel Type</em>'.
   * @see org.xtext.textualSlco.Channel#getChannelType()
   * @see #getChannel()
   * @generated
   */
  EAttribute getChannel_ChannelType();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.UnidirectionalChannel <em>Unidirectional Channel</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Unidirectional Channel</em>'.
   * @see org.xtext.textualSlco.UnidirectionalChannel
   * @generated
   */
  EClass getUnidirectionalChannel();

  /**
   * Returns the meta object for the reference '{@link org.xtext.textualSlco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the reference '<em>Source Object</em>'.
   * @see org.xtext.textualSlco.UnidirectionalChannel#getSourceObject()
   * @see #getUnidirectionalChannel()
   * @generated
   */
  EReference getUnidirectionalChannel_SourceObject();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Source Port</em>'.
   * @see org.xtext.textualSlco.UnidirectionalChannel#getSourcePort()
   * @see #getUnidirectionalChannel()
   * @generated
   */
  EReference getUnidirectionalChannel_SourcePort();

  /**
   * Returns the meta object for the reference '{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the reference '<em>Target Object</em>'.
   * @see org.xtext.textualSlco.UnidirectionalChannel#getTargetObject()
   * @see #getUnidirectionalChannel()
   * @generated
   */
  EReference getUnidirectionalChannel_TargetObject();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Target Port</em>'.
   * @see org.xtext.textualSlco.UnidirectionalChannel#getTargetPort()
   * @see #getUnidirectionalChannel()
   * @generated
   */
  EReference getUnidirectionalChannel_TargetPort();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.BidirectionalChannel <em>Bidirectional Channel</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Bidirectional Channel</em>'.
   * @see org.xtext.textualSlco.BidirectionalChannel
   * @generated
   */
  EClass getBidirectionalChannel();

  /**
   * Returns the meta object for the reference '{@link org.xtext.textualSlco.BidirectionalChannel#getObject1 <em>Object1</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the reference '<em>Object1</em>'.
   * @see org.xtext.textualSlco.BidirectionalChannel#getObject1()
   * @see #getBidirectionalChannel()
   * @generated
   */
  EReference getBidirectionalChannel_Object1();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.BidirectionalChannel#getPort1 <em>Port1</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Port1</em>'.
   * @see org.xtext.textualSlco.BidirectionalChannel#getPort1()
   * @see #getBidirectionalChannel()
   * @generated
   */
  EReference getBidirectionalChannel_Port1();

  /**
   * Returns the meta object for the reference '{@link org.xtext.textualSlco.BidirectionalChannel#getObject2 <em>Object2</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the reference '<em>Object2</em>'.
   * @see org.xtext.textualSlco.BidirectionalChannel#getObject2()
   * @see #getBidirectionalChannel()
   * @generated
   */
  EReference getBidirectionalChannel_Object2();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.BidirectionalChannel#getPort2 <em>Port2</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Port2</em>'.
   * @see org.xtext.textualSlco.BidirectionalChannel#getPort2()
   * @see #getBidirectionalChannel()
   * @generated
   */
  EReference getBidirectionalChannel_Port2();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.ArgumentType <em>Argument Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Argument Type</em>'.
   * @see org.xtext.textualSlco.ArgumentType
   * @generated
   */
  EClass getArgumentType();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.ArgumentType#getType <em>Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Type</em>'.
   * @see org.xtext.textualSlco.ArgumentType#getType()
   * @see #getArgumentType()
   * @generated
   */
  EAttribute getArgumentType_Type();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Variable <em>Variable</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Variable</em>'.
   * @see org.xtext.textualSlco.Variable
   * @generated
   */
  EClass getVariable();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Variable#getType <em>Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Type</em>'.
   * @see org.xtext.textualSlco.Variable#getType()
   * @see #getVariable()
   * @generated
   */
  EAttribute getVariable_Type();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.Variable#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.Variable#getName()
   * @see #getVariable()
   * @generated
   */
  EAttribute getVariable_Name();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.Variable#getIntialValue <em>Intial Value</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Intial Value</em>'.
   * @see org.xtext.textualSlco.Variable#getIntialValue()
   * @see #getVariable()
   * @generated
   */
  EReference getVariable_IntialValue();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.Expression <em>Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Expression</em>'.
   * @see org.xtext.textualSlco.Expression
   * @generated
   */
  EClass getExpression();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.BooleanConstantExpression <em>Boolean Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Boolean Constant Expression</em>'.
   * @see org.xtext.textualSlco.BooleanConstantExpression
   * @generated
   */
  EClass getBooleanConstantExpression();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.BooleanConstantExpression#getValue <em>Value</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Value</em>'.
   * @see org.xtext.textualSlco.BooleanConstantExpression#getValue()
   * @see #getBooleanConstantExpression()
   * @generated
   */
  EAttribute getBooleanConstantExpression_Value();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.IntegerConstantExpression <em>Integer Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Integer Constant Expression</em>'.
   * @see org.xtext.textualSlco.IntegerConstantExpression
   * @generated
   */
  EClass getIntegerConstantExpression();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.IntegerConstantExpression#getValue <em>Value</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Value</em>'.
   * @see org.xtext.textualSlco.IntegerConstantExpression#getValue()
   * @see #getIntegerConstantExpression()
   * @generated
   */
  EAttribute getIntegerConstantExpression_Value();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.StringConstantExpression <em>String Constant Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>String Constant Expression</em>'.
   * @see org.xtext.textualSlco.StringConstantExpression
   * @generated
   */
  EClass getStringConstantExpression();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.StringConstantExpression#getValue <em>Value</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Value</em>'.
   * @see org.xtext.textualSlco.StringConstantExpression#getValue()
   * @see #getStringConstantExpression()
   * @generated
   */
  EAttribute getStringConstantExpression_Value();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.VariableReference <em>Variable Reference</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Variable Reference</em>'.
   * @see org.xtext.textualSlco.VariableReference
   * @generated
   */
  EClass getVariableReference();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.VariableReference#getName <em>Name</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Name</em>'.
   * @see org.xtext.textualSlco.VariableReference#getName()
   * @see #getVariableReference()
   * @generated
   */
  EAttribute getVariableReference_Name();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.VariableExpression <em>Variable Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Variable Expression</em>'.
   * @see org.xtext.textualSlco.VariableExpression
   * @generated
   */
  EClass getVariableExpression();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.VariableExpression#getVariable <em>Variable</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Variable</em>'.
   * @see org.xtext.textualSlco.VariableExpression#getVariable()
   * @see #getVariableExpression()
   * @generated
   */
  EReference getVariableExpression_Variable();

  /**
   * Returns the meta object for class '{@link org.xtext.textualSlco.BinaryOperatorExpression <em>Binary Operator Expression</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for class '<em>Binary Operator Expression</em>'.
   * @see org.xtext.textualSlco.BinaryOperatorExpression
   * @generated
   */
  EClass getBinaryOperatorExpression();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Operand1</em>'.
   * @see org.xtext.textualSlco.BinaryOperatorExpression#getOperand1()
   * @see #getBinaryOperatorExpression()
   * @generated
   */
  EReference getBinaryOperatorExpression_Operand1();

  /**
   * Returns the meta object for the attribute '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperator <em>Operator</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the attribute '<em>Operator</em>'.
   * @see org.xtext.textualSlco.BinaryOperatorExpression#getOperator()
   * @see #getBinaryOperatorExpression()
   * @generated
   */
  EAttribute getBinaryOperatorExpression_Operator();

  /**
   * Returns the meta object for the containment reference '{@link org.xtext.textualSlco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for the containment reference '<em>Operand2</em>'.
   * @see org.xtext.textualSlco.BinaryOperatorExpression#getOperand2()
   * @see #getBinaryOperatorExpression()
   * @generated
   */
  EReference getBinaryOperatorExpression_Operand2();

  /**
   * Returns the meta object for enum '{@link org.xtext.textualSlco.PrimitiveType <em>Primitive Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for enum '<em>Primitive Type</em>'.
   * @see org.xtext.textualSlco.PrimitiveType
   * @generated
   */
  EEnum getPrimitiveType();

  /**
   * Returns the meta object for enum '{@link org.xtext.textualSlco.ChannelType <em>Channel Type</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for enum '<em>Channel Type</em>'.
   * @see org.xtext.textualSlco.ChannelType
   * @generated
   */
  EEnum getChannelType();

  /**
   * Returns the meta object for enum '{@link org.xtext.textualSlco.Operator <em>Operator</em>}'.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the meta object for enum '<em>Operator</em>'.
   * @see org.xtext.textualSlco.Operator
   * @generated
   */
  EEnum getOperator();

  /**
   * Returns the factory that creates the instances of the model.
   * <!-- begin-user-doc -->
   * <!-- end-user-doc -->
   * @return the factory that creates the instances of the model.
   * @generated
   */
  TextualSlcoFactory getTextualSlcoFactory();

  /**
   * <!-- begin-user-doc -->
   * Defines literals for the meta objects that represent
   * <ul>
   *   <li>each class,</li>
   *   <li>each feature of each class,</li>
   *   <li>each enum,</li>
   *   <li>and each data type</li>
   * </ul>
   * <!-- end-user-doc -->
   * @generated
   */
  interface Literals
  {
    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ModelImpl <em>Model</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ModelImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getModel()
     * @generated
     */
    EClass MODEL = eINSTANCE.getModel();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute MODEL__NAME = eINSTANCE.getModel_Name();

    /**
     * The meta object literal for the '<em><b>Classes</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference MODEL__CLASSES = eINSTANCE.getModel_Classes();

    /**
     * The meta object literal for the '<em><b>Objects</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference MODEL__OBJECTS = eINSTANCE.getModel_Objects();

    /**
     * The meta object literal for the '<em><b>Channels</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference MODEL__CHANNELS = eINSTANCE.getModel_Channels();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ClassImpl <em>Class</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ClassImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getClass_()
     * @generated
     */
    EClass CLASS = eINSTANCE.getClass_();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute CLASS__NAME = eINSTANCE.getClass_Name();

    /**
     * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference CLASS__VARIABLES = eINSTANCE.getClass_Variables();

    /**
     * The meta object literal for the '<em><b>Ports</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference CLASS__PORTS = eINSTANCE.getClass_Ports();

    /**
     * The meta object literal for the '<em><b>State Machines</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference CLASS__STATE_MACHINES = eINSTANCE.getClass_StateMachines();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.StateMachineImpl <em>State Machine</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.StateMachineImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStateMachine()
     * @generated
     */
    EClass STATE_MACHINE = eINSTANCE.getStateMachine();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute STATE_MACHINE__NAME = eINSTANCE.getStateMachine_Name();

    /**
     * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference STATE_MACHINE__VARIABLES = eINSTANCE.getStateMachine_Variables();

    /**
     * The meta object literal for the '<em><b>Vertices</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference STATE_MACHINE__VERTICES = eINSTANCE.getStateMachine_Vertices();

    /**
     * The meta object literal for the '<em><b>Transitions</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference STATE_MACHINE__TRANSITIONS = eINSTANCE.getStateMachine_Transitions();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.VertexImpl <em>Vertex</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.VertexImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVertex()
     * @generated
     */
    EClass VERTEX = eINSTANCE.getVertex();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute VERTEX__NAME = eINSTANCE.getVertex_Name();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.InitialImpl <em>Initial</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.InitialImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getInitial()
     * @generated
     */
    EClass INITIAL = eINSTANCE.getInitial();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.StateImpl <em>State</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.StateImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getState()
     * @generated
     */
    EClass STATE = eINSTANCE.getState();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.FinalImpl <em>Final</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.FinalImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getFinal()
     * @generated
     */
    EClass FINAL = eINSTANCE.getFinal();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.VertexReferenceImpl <em>Vertex Reference</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.VertexReferenceImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVertexReference()
     * @generated
     */
    EClass VERTEX_REFERENCE = eINSTANCE.getVertexReference();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute VERTEX_REFERENCE__NAME = eINSTANCE.getVertexReference_Name();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.TransitionImpl <em>Transition</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.TransitionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getTransition()
     * @generated
     */
    EClass TRANSITION = eINSTANCE.getTransition();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute TRANSITION__NAME = eINSTANCE.getTransition_Name();

    /**
     * The meta object literal for the '<em><b>Source</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference TRANSITION__SOURCE = eINSTANCE.getTransition_Source();

    /**
     * The meta object literal for the '<em><b>Target</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference TRANSITION__TARGET = eINSTANCE.getTransition_Target();

    /**
     * The meta object literal for the '<em><b>Statements</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference TRANSITION__STATEMENTS = eINSTANCE.getTransition_Statements();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.StatementImpl <em>Statement</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.StatementImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStatement()
     * @generated
     */
    EClass STATEMENT = eINSTANCE.getStatement();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.DelayImpl <em>Delay</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.DelayImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getDelay()
     * @generated
     */
    EClass DELAY = eINSTANCE.getDelay();

    /**
     * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute DELAY__VALUE = eINSTANCE.getDelay_Value();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.PortReferenceImpl <em>Port Reference</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.PortReferenceImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPortReference()
     * @generated
     */
    EClass PORT_REFERENCE = eINSTANCE.getPortReference();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute PORT_REFERENCE__NAME = eINSTANCE.getPortReference_Name();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.SignalReceptionImpl <em>Signal Reception</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.SignalReceptionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalReception()
     * @generated
     */
    EClass SIGNAL_RECEPTION = eINSTANCE.getSignalReception();

    /**
     * The meta object literal for the '<em><b>Signal Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute SIGNAL_RECEPTION__SIGNAL_NAME = eINSTANCE.getSignalReception_SignalName();

    /**
     * The meta object literal for the '<em><b>Arguments</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SIGNAL_RECEPTION__ARGUMENTS = eINSTANCE.getSignalReception_Arguments();

    /**
     * The meta object literal for the '<em><b>Condition</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SIGNAL_RECEPTION__CONDITION = eINSTANCE.getSignalReception_Condition();

    /**
     * The meta object literal for the '<em><b>Port</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SIGNAL_RECEPTION__PORT = eINSTANCE.getSignalReception_Port();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.SignalArgumentImpl <em>Signal Argument</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.SignalArgumentImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgument()
     * @generated
     */
    EClass SIGNAL_ARGUMENT = eINSTANCE.getSignalArgument();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.SignalArgumentVariableImpl <em>Signal Argument Variable</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.SignalArgumentVariableImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgumentVariable()
     * @generated
     */
    EClass SIGNAL_ARGUMENT_VARIABLE = eINSTANCE.getSignalArgumentVariable();

    /**
     * The meta object literal for the '<em><b>Variable</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SIGNAL_ARGUMENT_VARIABLE__VARIABLE = eINSTANCE.getSignalArgumentVariable_Variable();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.SignalArgumentExpressionImpl <em>Signal Argument Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.SignalArgumentExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSignalArgumentExpression()
     * @generated
     */
    EClass SIGNAL_ARGUMENT_EXPRESSION = eINSTANCE.getSignalArgumentExpression();

    /**
     * The meta object literal for the '<em><b>Expression</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SIGNAL_ARGUMENT_EXPRESSION__EXPRESSION = eINSTANCE.getSignalArgumentExpression_Expression();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.AssignmentImpl <em>Assignment</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.AssignmentImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getAssignment()
     * @generated
     */
    EClass ASSIGNMENT = eINSTANCE.getAssignment();

    /**
     * The meta object literal for the '<em><b>Variable</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference ASSIGNMENT__VARIABLE = eINSTANCE.getAssignment_Variable();

    /**
     * The meta object literal for the '<em><b>Expression</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference ASSIGNMENT__EXPRESSION = eINSTANCE.getAssignment_Expression();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.SendSignalImpl <em>Send Signal</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.SendSignalImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getSendSignal()
     * @generated
     */
    EClass SEND_SIGNAL = eINSTANCE.getSendSignal();

    /**
     * The meta object literal for the '<em><b>Signal Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute SEND_SIGNAL__SIGNAL_NAME = eINSTANCE.getSendSignal_SignalName();

    /**
     * The meta object literal for the '<em><b>Arguments</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SEND_SIGNAL__ARGUMENTS = eINSTANCE.getSendSignal_Arguments();

    /**
     * The meta object literal for the '<em><b>Port</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference SEND_SIGNAL__PORT = eINSTANCE.getSendSignal_Port();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ObjectImpl <em>Object</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ObjectImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getObject()
     * @generated
     */
    EClass OBJECT = eINSTANCE.getObject();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute OBJECT__NAME = eINSTANCE.getObject_Name();

    /**
     * The meta object literal for the '<em><b>Class</b></em>' reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference OBJECT__CLASS = eINSTANCE.getObject_Class();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.PortImpl <em>Port</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.PortImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPort()
     * @generated
     */
    EClass PORT = eINSTANCE.getPort();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute PORT__NAME = eINSTANCE.getPort_Name();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ChannelImpl <em>Channel</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ChannelImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getChannel()
     * @generated
     */
    EClass CHANNEL = eINSTANCE.getChannel();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute CHANNEL__NAME = eINSTANCE.getChannel_Name();

    /**
     * The meta object literal for the '<em><b>Argument Types</b></em>' containment reference list feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference CHANNEL__ARGUMENT_TYPES = eINSTANCE.getChannel_ArgumentTypes();

    /**
     * The meta object literal for the '<em><b>Channel Type</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute CHANNEL__CHANNEL_TYPE = eINSTANCE.getChannel_ChannelType();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.UnidirectionalChannelImpl <em>Unidirectional Channel</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.UnidirectionalChannelImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getUnidirectionalChannel()
     * @generated
     */
    EClass UNIDIRECTIONAL_CHANNEL = eINSTANCE.getUnidirectionalChannel();

    /**
     * The meta object literal for the '<em><b>Source Object</b></em>' reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT = eINSTANCE.getUnidirectionalChannel_SourceObject();

    /**
     * The meta object literal for the '<em><b>Source Port</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference UNIDIRECTIONAL_CHANNEL__SOURCE_PORT = eINSTANCE.getUnidirectionalChannel_SourcePort();

    /**
     * The meta object literal for the '<em><b>Target Object</b></em>' reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT = eINSTANCE.getUnidirectionalChannel_TargetObject();

    /**
     * The meta object literal for the '<em><b>Target Port</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference UNIDIRECTIONAL_CHANNEL__TARGET_PORT = eINSTANCE.getUnidirectionalChannel_TargetPort();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.BidirectionalChannelImpl <em>Bidirectional Channel</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.BidirectionalChannelImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBidirectionalChannel()
     * @generated
     */
    EClass BIDIRECTIONAL_CHANNEL = eINSTANCE.getBidirectionalChannel();

    /**
     * The meta object literal for the '<em><b>Object1</b></em>' reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BIDIRECTIONAL_CHANNEL__OBJECT1 = eINSTANCE.getBidirectionalChannel_Object1();

    /**
     * The meta object literal for the '<em><b>Port1</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BIDIRECTIONAL_CHANNEL__PORT1 = eINSTANCE.getBidirectionalChannel_Port1();

    /**
     * The meta object literal for the '<em><b>Object2</b></em>' reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BIDIRECTIONAL_CHANNEL__OBJECT2 = eINSTANCE.getBidirectionalChannel_Object2();

    /**
     * The meta object literal for the '<em><b>Port2</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BIDIRECTIONAL_CHANNEL__PORT2 = eINSTANCE.getBidirectionalChannel_Port2();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ArgumentTypeImpl <em>Argument Type</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ArgumentTypeImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getArgumentType()
     * @generated
     */
    EClass ARGUMENT_TYPE = eINSTANCE.getArgumentType();

    /**
     * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute ARGUMENT_TYPE__TYPE = eINSTANCE.getArgumentType_Type();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.VariableImpl <em>Variable</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.VariableImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariable()
     * @generated
     */
    EClass VARIABLE = eINSTANCE.getVariable();

    /**
     * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute VARIABLE__TYPE = eINSTANCE.getVariable_Type();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute VARIABLE__NAME = eINSTANCE.getVariable_Name();

    /**
     * The meta object literal for the '<em><b>Intial Value</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference VARIABLE__INTIAL_VALUE = eINSTANCE.getVariable_IntialValue();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.ExpressionImpl <em>Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.ExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getExpression()
     * @generated
     */
    EClass EXPRESSION = eINSTANCE.getExpression();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.BooleanConstantExpressionImpl <em>Boolean Constant Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.BooleanConstantExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBooleanConstantExpression()
     * @generated
     */
    EClass BOOLEAN_CONSTANT_EXPRESSION = eINSTANCE.getBooleanConstantExpression();

    /**
     * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute BOOLEAN_CONSTANT_EXPRESSION__VALUE = eINSTANCE.getBooleanConstantExpression_Value();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.IntegerConstantExpressionImpl <em>Integer Constant Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.IntegerConstantExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getIntegerConstantExpression()
     * @generated
     */
    EClass INTEGER_CONSTANT_EXPRESSION = eINSTANCE.getIntegerConstantExpression();

    /**
     * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute INTEGER_CONSTANT_EXPRESSION__VALUE = eINSTANCE.getIntegerConstantExpression_Value();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.StringConstantExpressionImpl <em>String Constant Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.StringConstantExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getStringConstantExpression()
     * @generated
     */
    EClass STRING_CONSTANT_EXPRESSION = eINSTANCE.getStringConstantExpression();

    /**
     * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute STRING_CONSTANT_EXPRESSION__VALUE = eINSTANCE.getStringConstantExpression_Value();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.VariableReferenceImpl <em>Variable Reference</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.VariableReferenceImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariableReference()
     * @generated
     */
    EClass VARIABLE_REFERENCE = eINSTANCE.getVariableReference();

    /**
     * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute VARIABLE_REFERENCE__NAME = eINSTANCE.getVariableReference_Name();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.VariableExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getVariableExpression()
     * @generated
     */
    EClass VARIABLE_EXPRESSION = eINSTANCE.getVariableExpression();

    /**
     * The meta object literal for the '<em><b>Variable</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference VARIABLE_EXPRESSION__VARIABLE = eINSTANCE.getVariableExpression_Variable();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.impl.BinaryOperatorExpressionImpl <em>Binary Operator Expression</em>}' class.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.impl.BinaryOperatorExpressionImpl
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getBinaryOperatorExpression()
     * @generated
     */
    EClass BINARY_OPERATOR_EXPRESSION = eINSTANCE.getBinaryOperatorExpression();

    /**
     * The meta object literal for the '<em><b>Operand1</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BINARY_OPERATOR_EXPRESSION__OPERAND1 = eINSTANCE.getBinaryOperatorExpression_Operand1();

    /**
     * The meta object literal for the '<em><b>Operator</b></em>' attribute feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EAttribute BINARY_OPERATOR_EXPRESSION__OPERATOR = eINSTANCE.getBinaryOperatorExpression_Operator();

    /**
     * The meta object literal for the '<em><b>Operand2</b></em>' containment reference feature.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @generated
     */
    EReference BINARY_OPERATOR_EXPRESSION__OPERAND2 = eINSTANCE.getBinaryOperatorExpression_Operand2();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.PrimitiveType <em>Primitive Type</em>}' enum.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.PrimitiveType
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getPrimitiveType()
     * @generated
     */
    EEnum PRIMITIVE_TYPE = eINSTANCE.getPrimitiveType();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.ChannelType <em>Channel Type</em>}' enum.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.ChannelType
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getChannelType()
     * @generated
     */
    EEnum CHANNEL_TYPE = eINSTANCE.getChannelType();

    /**
     * The meta object literal for the '{@link org.xtext.textualSlco.Operator <em>Operator</em>}' enum.
     * <!-- begin-user-doc -->
     * <!-- end-user-doc -->
     * @see org.xtext.textualSlco.Operator
     * @see org.xtext.textualSlco.impl.TextualSlcoPackageImpl#getOperator()
     * @generated
     */
    EEnum OPERATOR = eINSTANCE.getOperator();

  }

} //TextualSlcoPackage

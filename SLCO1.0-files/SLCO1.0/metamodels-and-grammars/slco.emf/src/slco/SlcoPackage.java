/**
 */
package slco;

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
 * @see slco.SlcoFactory
 * @model kind="package"
 * @generated
 */
public interface SlcoPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "slco";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "slco";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "slco";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	SlcoPackage eINSTANCE = slco.impl.SlcoPackageImpl.init();

	/**
	 * The meta object id for the '{@link slco.impl.ArgumentTypeImpl <em>Argument Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ArgumentTypeImpl
	 * @see slco.impl.SlcoPackageImpl#getArgumentType()
	 * @generated
	 */
	int ARGUMENT_TYPE = 0;

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
	 * The meta object id for the '{@link slco.impl.StatementImpl <em>Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.StatementImpl
	 * @see slco.impl.SlcoPackageImpl#getStatement()
	 * @generated
	 */
	int STATEMENT = 23;

	/**
	 * The number of structural features of the '<em>Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATEMENT_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link slco.impl.AssignmentImpl <em>Assignment</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.AssignmentImpl
	 * @see slco.impl.SlcoPackageImpl#getAssignment()
	 * @generated
	 */
	int ASSIGNMENT = 1;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
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
	 * The meta object id for the '{@link slco.impl.ChannelImpl <em>Channel</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ChannelImpl
	 * @see slco.impl.SlcoPackageImpl#getChannel()
	 * @generated
	 */
	int CHANNEL = 5;

	/**
	 * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__ARGUMENT_TYPES = 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CHANNEL__NAME = 1;

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
	 * The meta object id for the '{@link slco.impl.BidirectionalChannelImpl <em>Bidirectional Channel</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.BidirectionalChannelImpl
	 * @see slco.impl.SlcoPackageImpl#getBidirectionalChannel()
	 * @generated
	 */
	int BIDIRECTIONAL_CHANNEL = 2;

	/**
	 * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__ARGUMENT_TYPES = CHANNEL__ARGUMENT_TYPES;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__NAME = CHANNEL__NAME;

	/**
	 * The feature id for the '<em><b>Channel Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__CHANNEL_TYPE = CHANNEL__CHANNEL_TYPE;

	/**
	 * The feature id for the '<em><b>Port1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__PORT1 = CHANNEL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Port2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__PORT2 = CHANNEL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Object1</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__OBJECT1 = CHANNEL_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Object2</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL__OBJECT2 = CHANNEL_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Bidirectional Channel</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BIDIRECTIONAL_CHANNEL_FEATURE_COUNT = CHANNEL_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link slco.impl.ExpressionImpl <em>Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getExpression()
	 * @generated
	 */
	int EXPRESSION = 9;

	/**
	 * The number of structural features of the '<em>Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int EXPRESSION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link slco.impl.BinaryOperatorExpressionImpl <em>Binary Operator Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.BinaryOperatorExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getBinaryOperatorExpression()
	 * @generated
	 */
	int BINARY_OPERATOR_EXPRESSION = 3;

	/**
	 * The feature id for the '<em><b>Operator</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_OPERATOR_EXPRESSION__OPERATOR = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Operand1</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BINARY_OPERATOR_EXPRESSION__OPERAND1 = EXPRESSION_FEATURE_COUNT + 1;

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
	 * The meta object id for the '{@link slco.impl.ConstantExpressionImpl <em>Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ConstantExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getConstantExpression()
	 * @generated
	 */
	int CONSTANT_EXPRESSION = 7;

	/**
	 * The number of structural features of the '<em>Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONSTANT_EXPRESSION_FEATURE_COUNT = EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link slco.impl.BooleanConstantExpressionImpl <em>Boolean Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.BooleanConstantExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getBooleanConstantExpression()
	 * @generated
	 */
	int BOOLEAN_CONSTANT_EXPRESSION = 4;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BOOLEAN_CONSTANT_EXPRESSION__VALUE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Boolean Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int BOOLEAN_CONSTANT_EXPRESSION_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link slco.impl.ClassImpl <em>Class</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ClassImpl
	 * @see slco.impl.SlcoPackageImpl#getClass_()
	 * @generated
	 */
	int CLASS = 6;

	/**
	 * The feature id for the '<em><b>State Machines</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS__STATE_MACHINES = 0;

	/**
	 * The feature id for the '<em><b>Ports</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS__PORTS = 1;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS__VARIABLES = 2;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS__NAME = 3;

	/**
	 * The number of structural features of the '<em>Class</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link slco.impl.DelayImpl <em>Delay</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.DelayImpl
	 * @see slco.impl.SlcoPackageImpl#getDelay()
	 * @generated
	 */
	int DELAY = 8;

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
	 * The meta object id for the '{@link slco.impl.VertexImpl <em>Vertex</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.VertexImpl
	 * @see slco.impl.SlcoPackageImpl#getVertex()
	 * @generated
	 */
	int VERTEX = 30;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VERTEX__NAME = 0;

	/**
	 * The feature id for the '<em><b>Incoming</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VERTEX__INCOMING = 1;

	/**
	 * The feature id for the '<em><b>Outgoing</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VERTEX__OUTGOING = 2;

	/**
	 * The number of structural features of the '<em>Vertex</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VERTEX_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link slco.impl.FinalImpl <em>Final</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.FinalImpl
	 * @see slco.impl.SlcoPackageImpl#getFinal()
	 * @generated
	 */
	int FINAL = 10;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FINAL__NAME = VERTEX__NAME;

	/**
	 * The feature id for the '<em><b>Incoming</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FINAL__INCOMING = VERTEX__INCOMING;

	/**
	 * The feature id for the '<em><b>Outgoing</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FINAL__OUTGOING = VERTEX__OUTGOING;

	/**
	 * The number of structural features of the '<em>Final</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FINAL_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link slco.impl.InitialImpl <em>Initial</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.InitialImpl
	 * @see slco.impl.SlcoPackageImpl#getInitial()
	 * @generated
	 */
	int INITIAL = 11;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INITIAL__NAME = VERTEX__NAME;

	/**
	 * The feature id for the '<em><b>Incoming</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INITIAL__INCOMING = VERTEX__INCOMING;

	/**
	 * The feature id for the '<em><b>Outgoing</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INITIAL__OUTGOING = VERTEX__OUTGOING;

	/**
	 * The number of structural features of the '<em>Initial</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INITIAL_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link slco.impl.IntegerConstantExpressionImpl <em>Integer Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.IntegerConstantExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getIntegerConstantExpression()
	 * @generated
	 */
	int INTEGER_CONSTANT_EXPRESSION = 12;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTEGER_CONSTANT_EXPRESSION__VALUE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Integer Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTEGER_CONSTANT_EXPRESSION_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link slco.impl.ModelImpl <em>Model</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ModelImpl
	 * @see slco.impl.SlcoPackageImpl#getModel()
	 * @generated
	 */
	int MODEL = 13;

	/**
	 * The feature id for the '<em><b>Classes</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__CLASSES = 0;

	/**
	 * The feature id for the '<em><b>Objects</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__OBJECTS = 1;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MODEL__NAME = 2;

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
	 * The meta object id for the '{@link slco.impl.ObjectImpl <em>Object</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.ObjectImpl
	 * @see slco.impl.SlcoPackageImpl#getObject()
	 * @generated
	 */
	int OBJECT = 14;

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
	 * The meta object id for the '{@link slco.impl.PortImpl <em>Port</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.PortImpl
	 * @see slco.impl.SlcoPackageImpl#getPort()
	 * @generated
	 */
	int PORT = 15;

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
	 * The meta object id for the '{@link slco.impl.SendSignalImpl <em>Send Signal</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.SendSignalImpl
	 * @see slco.impl.SlcoPackageImpl#getSendSignal()
	 * @generated
	 */
	int SEND_SIGNAL = 16;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND_SIGNAL__PORT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND_SIGNAL__SIGNAL_NAME = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Arguments</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND_SIGNAL__ARGUMENTS = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Send Signal</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SEND_SIGNAL_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The meta object id for the '{@link slco.impl.SignalArgumentImpl <em>Signal Argument</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.SignalArgumentImpl
	 * @see slco.impl.SlcoPackageImpl#getSignalArgument()
	 * @generated
	 */
	int SIGNAL_ARGUMENT = 17;

	/**
	 * The number of structural features of the '<em>Signal Argument</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_ARGUMENT_FEATURE_COUNT = 0;

	/**
	 * The meta object id for the '{@link slco.impl.SignalArgumentExpressionImpl <em>Signal Argument Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.SignalArgumentExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getSignalArgumentExpression()
	 * @generated
	 */
	int SIGNAL_ARGUMENT_EXPRESSION = 18;

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
	 * The meta object id for the '{@link slco.impl.SignalArgumentVariableImpl <em>Signal Argument Variable</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.SignalArgumentVariableImpl
	 * @see slco.impl.SlcoPackageImpl#getSignalArgumentVariable()
	 * @generated
	 */
	int SIGNAL_ARGUMENT_VARIABLE = 19;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
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
	 * The meta object id for the '{@link slco.impl.SignalReceptionImpl <em>Signal Reception</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.SignalReceptionImpl
	 * @see slco.impl.SlcoPackageImpl#getSignalReception()
	 * @generated
	 */
	int SIGNAL_RECEPTION = 20;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_RECEPTION__PORT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_RECEPTION__SIGNAL_NAME = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Arguments</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_RECEPTION__ARGUMENTS = STATEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Condition</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_RECEPTION__CONDITION = STATEMENT_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Signal Reception</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL_RECEPTION_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link slco.impl.StateImpl <em>State</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.StateImpl
	 * @see slco.impl.SlcoPackageImpl#getState()
	 * @generated
	 */
	int STATE = 21;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE__NAME = VERTEX__NAME;

	/**
	 * The feature id for the '<em><b>Incoming</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE__INCOMING = VERTEX__INCOMING;

	/**
	 * The feature id for the '<em><b>Outgoing</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE__OUTGOING = VERTEX__OUTGOING;

	/**
	 * The number of structural features of the '<em>State</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_FEATURE_COUNT = VERTEX_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link slco.impl.StateMachineImpl <em>State Machine</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.StateMachineImpl
	 * @see slco.impl.SlcoPackageImpl#getStateMachine()
	 * @generated
	 */
	int STATE_MACHINE = 22;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_MACHINE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Vertices</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_MACHINE__VERTICES = 1;

	/**
	 * The feature id for the '<em><b>Transitions</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_MACHINE__TRANSITIONS = 2;

	/**
	 * The feature id for the '<em><b>Variables</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_MACHINE__VARIABLES = 3;

	/**
	 * The number of structural features of the '<em>State Machine</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STATE_MACHINE_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link slco.impl.StringConstantExpressionImpl <em>String Constant Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.StringConstantExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getStringConstantExpression()
	 * @generated
	 */
	int STRING_CONSTANT_EXPRESSION = 24;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STRING_CONSTANT_EXPRESSION__VALUE = CONSTANT_EXPRESSION_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>String Constant Expression</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int STRING_CONSTANT_EXPRESSION_FEATURE_COUNT = CONSTANT_EXPRESSION_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link slco.impl.TextualStatementImpl <em>Textual Statement</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.TextualStatementImpl
	 * @see slco.impl.SlcoPackageImpl#getTextualStatement()
	 * @generated
	 */
	int TEXTUAL_STATEMENT = 25;

	/**
	 * The feature id for the '<em><b>Statement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TEXTUAL_STATEMENT__STATEMENT = STATEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Textual Statement</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TEXTUAL_STATEMENT_FEATURE_COUNT = STATEMENT_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link slco.impl.TransitionImpl <em>Transition</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.TransitionImpl
	 * @see slco.impl.SlcoPackageImpl#getTransition()
	 * @generated
	 */
	int TRANSITION = 26;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION__NAME = 0;

	/**
	 * The feature id for the '<em><b>Source</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TRANSITION__SOURCE = 1;

	/**
	 * The feature id for the '<em><b>Target</b></em>' reference.
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
	 * The meta object id for the '{@link slco.impl.UnidirectionalChannelImpl <em>Unidirectional Channel</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.UnidirectionalChannelImpl
	 * @see slco.impl.SlcoPackageImpl#getUnidirectionalChannel()
	 * @generated
	 */
	int UNIDIRECTIONAL_CHANNEL = 27;

	/**
	 * The feature id for the '<em><b>Argument Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__ARGUMENT_TYPES = CHANNEL__ARGUMENT_TYPES;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__NAME = CHANNEL__NAME;

	/**
	 * The feature id for the '<em><b>Channel Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__CHANNEL_TYPE = CHANNEL__CHANNEL_TYPE;

	/**
	 * The feature id for the '<em><b>Source Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__SOURCE_PORT = CHANNEL_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Target Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__TARGET_PORT = CHANNEL_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Source Object</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT = CHANNEL_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Target Object</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT = CHANNEL_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Unidirectional Channel</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int UNIDIRECTIONAL_CHANNEL_FEATURE_COUNT = CHANNEL_FEATURE_COUNT + 4;

	/**
	 * The meta object id for the '{@link slco.impl.VariableImpl <em>Variable</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.VariableImpl
	 * @see slco.impl.SlcoPackageImpl#getVariable()
	 * @generated
	 */
	int VARIABLE = 28;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__NAME = 0;

	/**
	 * The feature id for the '<em><b>Initial Value</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__INITIAL_VALUE = 1;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE__TYPE = 2;

	/**
	 * The number of structural features of the '<em>Variable</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int VARIABLE_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link slco.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.impl.VariableExpressionImpl
	 * @see slco.impl.SlcoPackageImpl#getVariableExpression()
	 * @generated
	 */
	int VARIABLE_EXPRESSION = 29;

	/**
	 * The feature id for the '<em><b>Variable</b></em>' reference.
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
	 * The meta object id for the '{@link slco.ChannelTypeEnum <em>Channel Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.ChannelTypeEnum
	 * @see slco.impl.SlcoPackageImpl#getChannelTypeEnum()
	 * @generated
	 */
	int CHANNEL_TYPE_ENUM = 31;

	/**
	 * The meta object id for the '{@link slco.OperatorEnum <em>Operator Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.OperatorEnum
	 * @see slco.impl.SlcoPackageImpl#getOperatorEnum()
	 * @generated
	 */
	int OPERATOR_ENUM = 32;

	/**
	 * The meta object id for the '{@link slco.PrimitiveTypeEnum <em>Primitive Type Enum</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco.PrimitiveTypeEnum
	 * @see slco.impl.SlcoPackageImpl#getPrimitiveTypeEnum()
	 * @generated
	 */
	int PRIMITIVE_TYPE_ENUM = 33;


	/**
	 * Returns the meta object for class '{@link slco.ArgumentType <em>Argument Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Argument Type</em>'.
	 * @see slco.ArgumentType
	 * @generated
	 */
	EClass getArgumentType();

	/**
	 * Returns the meta object for the attribute '{@link slco.ArgumentType#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see slco.ArgumentType#getType()
	 * @see #getArgumentType()
	 * @generated
	 */
	EAttribute getArgumentType_Type();

	/**
	 * Returns the meta object for class '{@link slco.Assignment <em>Assignment</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Assignment</em>'.
	 * @see slco.Assignment
	 * @generated
	 */
	EClass getAssignment();

	/**
	 * Returns the meta object for the reference '{@link slco.Assignment#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Variable</em>'.
	 * @see slco.Assignment#getVariable()
	 * @see #getAssignment()
	 * @generated
	 */
	EReference getAssignment_Variable();

	/**
	 * Returns the meta object for the containment reference '{@link slco.Assignment#getExpression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expression</em>'.
	 * @see slco.Assignment#getExpression()
	 * @see #getAssignment()
	 * @generated
	 */
	EReference getAssignment_Expression();

	/**
	 * Returns the meta object for class '{@link slco.BidirectionalChannel <em>Bidirectional Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Bidirectional Channel</em>'.
	 * @see slco.BidirectionalChannel
	 * @generated
	 */
	EClass getBidirectionalChannel();

	/**
	 * Returns the meta object for the reference '{@link slco.BidirectionalChannel#getPort1 <em>Port1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port1</em>'.
	 * @see slco.BidirectionalChannel#getPort1()
	 * @see #getBidirectionalChannel()
	 * @generated
	 */
	EReference getBidirectionalChannel_Port1();

	/**
	 * Returns the meta object for the reference '{@link slco.BidirectionalChannel#getPort2 <em>Port2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port2</em>'.
	 * @see slco.BidirectionalChannel#getPort2()
	 * @see #getBidirectionalChannel()
	 * @generated
	 */
	EReference getBidirectionalChannel_Port2();

	/**
	 * Returns the meta object for the reference '{@link slco.BidirectionalChannel#getObject1 <em>Object1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Object1</em>'.
	 * @see slco.BidirectionalChannel#getObject1()
	 * @see #getBidirectionalChannel()
	 * @generated
	 */
	EReference getBidirectionalChannel_Object1();

	/**
	 * Returns the meta object for the reference '{@link slco.BidirectionalChannel#getObject2 <em>Object2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Object2</em>'.
	 * @see slco.BidirectionalChannel#getObject2()
	 * @see #getBidirectionalChannel()
	 * @generated
	 */
	EReference getBidirectionalChannel_Object2();

	/**
	 * Returns the meta object for class '{@link slco.BinaryOperatorExpression <em>Binary Operator Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Binary Operator Expression</em>'.
	 * @see slco.BinaryOperatorExpression
	 * @generated
	 */
	EClass getBinaryOperatorExpression();

	/**
	 * Returns the meta object for the attribute '{@link slco.BinaryOperatorExpression#getOperator <em>Operator</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Operator</em>'.
	 * @see slco.BinaryOperatorExpression#getOperator()
	 * @see #getBinaryOperatorExpression()
	 * @generated
	 */
	EAttribute getBinaryOperatorExpression_Operator();

	/**
	 * Returns the meta object for the containment reference '{@link slco.BinaryOperatorExpression#getOperand1 <em>Operand1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand1</em>'.
	 * @see slco.BinaryOperatorExpression#getOperand1()
	 * @see #getBinaryOperatorExpression()
	 * @generated
	 */
	EReference getBinaryOperatorExpression_Operand1();

	/**
	 * Returns the meta object for the containment reference '{@link slco.BinaryOperatorExpression#getOperand2 <em>Operand2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Operand2</em>'.
	 * @see slco.BinaryOperatorExpression#getOperand2()
	 * @see #getBinaryOperatorExpression()
	 * @generated
	 */
	EReference getBinaryOperatorExpression_Operand2();

	/**
	 * Returns the meta object for class '{@link slco.BooleanConstantExpression <em>Boolean Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Boolean Constant Expression</em>'.
	 * @see slco.BooleanConstantExpression
	 * @generated
	 */
	EClass getBooleanConstantExpression();

	/**
	 * Returns the meta object for the attribute '{@link slco.BooleanConstantExpression#isValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see slco.BooleanConstantExpression#isValue()
	 * @see #getBooleanConstantExpression()
	 * @generated
	 */
	EAttribute getBooleanConstantExpression_Value();

	/**
	 * Returns the meta object for class '{@link slco.Channel <em>Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Channel</em>'.
	 * @see slco.Channel
	 * @generated
	 */
	EClass getChannel();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Channel#getArgumentTypes <em>Argument Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Argument Types</em>'.
	 * @see slco.Channel#getArgumentTypes()
	 * @see #getChannel()
	 * @generated
	 */
	EReference getChannel_ArgumentTypes();

	/**
	 * Returns the meta object for the attribute '{@link slco.Channel#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Channel#getName()
	 * @see #getChannel()
	 * @generated
	 */
	EAttribute getChannel_Name();

	/**
	 * Returns the meta object for the attribute '{@link slco.Channel#getChannelType <em>Channel Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Channel Type</em>'.
	 * @see slco.Channel#getChannelType()
	 * @see #getChannel()
	 * @generated
	 */
	EAttribute getChannel_ChannelType();

	/**
	 * Returns the meta object for class '{@link slco.Class <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Class</em>'.
	 * @see slco.Class
	 * @generated
	 */
	EClass getClass_();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Class#getStateMachines <em>State Machines</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>State Machines</em>'.
	 * @see slco.Class#getStateMachines()
	 * @see #getClass_()
	 * @generated
	 */
	EReference getClass_StateMachines();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Class#getPorts <em>Ports</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Ports</em>'.
	 * @see slco.Class#getPorts()
	 * @see #getClass_()
	 * @generated
	 */
	EReference getClass_Ports();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Class#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see slco.Class#getVariables()
	 * @see #getClass_()
	 * @generated
	 */
	EReference getClass_Variables();

	/**
	 * Returns the meta object for the attribute '{@link slco.Class#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Class#getName()
	 * @see #getClass_()
	 * @generated
	 */
	EAttribute getClass_Name();

	/**
	 * Returns the meta object for class '{@link slco.ConstantExpression <em>Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Constant Expression</em>'.
	 * @see slco.ConstantExpression
	 * @generated
	 */
	EClass getConstantExpression();

	/**
	 * Returns the meta object for class '{@link slco.Delay <em>Delay</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Delay</em>'.
	 * @see slco.Delay
	 * @generated
	 */
	EClass getDelay();

	/**
	 * Returns the meta object for the attribute '{@link slco.Delay#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see slco.Delay#getValue()
	 * @see #getDelay()
	 * @generated
	 */
	EAttribute getDelay_Value();

	/**
	 * Returns the meta object for class '{@link slco.Expression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Expression</em>'.
	 * @see slco.Expression
	 * @generated
	 */
	EClass getExpression();

	/**
	 * Returns the meta object for class '{@link slco.Final <em>Final</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Final</em>'.
	 * @see slco.Final
	 * @generated
	 */
	EClass getFinal();

	/**
	 * Returns the meta object for class '{@link slco.Initial <em>Initial</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Initial</em>'.
	 * @see slco.Initial
	 * @generated
	 */
	EClass getInitial();

	/**
	 * Returns the meta object for class '{@link slco.IntegerConstantExpression <em>Integer Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Integer Constant Expression</em>'.
	 * @see slco.IntegerConstantExpression
	 * @generated
	 */
	EClass getIntegerConstantExpression();

	/**
	 * Returns the meta object for the attribute '{@link slco.IntegerConstantExpression#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see slco.IntegerConstantExpression#getValue()
	 * @see #getIntegerConstantExpression()
	 * @generated
	 */
	EAttribute getIntegerConstantExpression_Value();

	/**
	 * Returns the meta object for class '{@link slco.Model <em>Model</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Model</em>'.
	 * @see slco.Model
	 * @generated
	 */
	EClass getModel();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Model#getClasses <em>Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Classes</em>'.
	 * @see slco.Model#getClasses()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_Classes();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Model#getObjects <em>Objects</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Objects</em>'.
	 * @see slco.Model#getObjects()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_Objects();

	/**
	 * Returns the meta object for the attribute '{@link slco.Model#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Model#getName()
	 * @see #getModel()
	 * @generated
	 */
	EAttribute getModel_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Model#getChannels <em>Channels</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Channels</em>'.
	 * @see slco.Model#getChannels()
	 * @see #getModel()
	 * @generated
	 */
	EReference getModel_Channels();

	/**
	 * Returns the meta object for class '{@link slco.Object <em>Object</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Object</em>'.
	 * @see slco.Object
	 * @generated
	 */
	EClass getObject();

	/**
	 * Returns the meta object for the attribute '{@link slco.Object#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Object#getName()
	 * @see #getObject()
	 * @generated
	 */
	EAttribute getObject_Name();

	/**
	 * Returns the meta object for the reference '{@link slco.Object#getClass_ <em>Class</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Class</em>'.
	 * @see slco.Object#getClass_()
	 * @see #getObject()
	 * @generated
	 */
	EReference getObject_Class();

	/**
	 * Returns the meta object for class '{@link slco.Port <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Port</em>'.
	 * @see slco.Port
	 * @generated
	 */
	EClass getPort();

	/**
	 * Returns the meta object for the attribute '{@link slco.Port#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Port#getName()
	 * @see #getPort()
	 * @generated
	 */
	EAttribute getPort_Name();

	/**
	 * Returns the meta object for class '{@link slco.SendSignal <em>Send Signal</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Send Signal</em>'.
	 * @see slco.SendSignal
	 * @generated
	 */
	EClass getSendSignal();

	/**
	 * Returns the meta object for the reference '{@link slco.SendSignal#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco.SendSignal#getPort()
	 * @see #getSendSignal()
	 * @generated
	 */
	EReference getSendSignal_Port();

	/**
	 * Returns the meta object for the attribute '{@link slco.SendSignal#getSignalName <em>Signal Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Signal Name</em>'.
	 * @see slco.SendSignal#getSignalName()
	 * @see #getSendSignal()
	 * @generated
	 */
	EAttribute getSendSignal_SignalName();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.SendSignal#getArguments <em>Arguments</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Arguments</em>'.
	 * @see slco.SendSignal#getArguments()
	 * @see #getSendSignal()
	 * @generated
	 */
	EReference getSendSignal_Arguments();

	/**
	 * Returns the meta object for class '{@link slco.SignalArgument <em>Signal Argument</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal Argument</em>'.
	 * @see slco.SignalArgument
	 * @generated
	 */
	EClass getSignalArgument();

	/**
	 * Returns the meta object for class '{@link slco.SignalArgumentExpression <em>Signal Argument Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal Argument Expression</em>'.
	 * @see slco.SignalArgumentExpression
	 * @generated
	 */
	EClass getSignalArgumentExpression();

	/**
	 * Returns the meta object for the containment reference '{@link slco.SignalArgumentExpression#getExpression <em>Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Expression</em>'.
	 * @see slco.SignalArgumentExpression#getExpression()
	 * @see #getSignalArgumentExpression()
	 * @generated
	 */
	EReference getSignalArgumentExpression_Expression();

	/**
	 * Returns the meta object for class '{@link slco.SignalArgumentVariable <em>Signal Argument Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal Argument Variable</em>'.
	 * @see slco.SignalArgumentVariable
	 * @generated
	 */
	EClass getSignalArgumentVariable();

	/**
	 * Returns the meta object for the reference '{@link slco.SignalArgumentVariable#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Variable</em>'.
	 * @see slco.SignalArgumentVariable#getVariable()
	 * @see #getSignalArgumentVariable()
	 * @generated
	 */
	EReference getSignalArgumentVariable_Variable();

	/**
	 * Returns the meta object for class '{@link slco.SignalReception <em>Signal Reception</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal Reception</em>'.
	 * @see slco.SignalReception
	 * @generated
	 */
	EClass getSignalReception();

	/**
	 * Returns the meta object for the reference '{@link slco.SignalReception#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco.SignalReception#getPort()
	 * @see #getSignalReception()
	 * @generated
	 */
	EReference getSignalReception_Port();

	/**
	 * Returns the meta object for the attribute '{@link slco.SignalReception#getSignalName <em>Signal Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Signal Name</em>'.
	 * @see slco.SignalReception#getSignalName()
	 * @see #getSignalReception()
	 * @generated
	 */
	EAttribute getSignalReception_SignalName();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.SignalReception#getArguments <em>Arguments</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Arguments</em>'.
	 * @see slco.SignalReception#getArguments()
	 * @see #getSignalReception()
	 * @generated
	 */
	EReference getSignalReception_Arguments();

	/**
	 * Returns the meta object for the containment reference '{@link slco.SignalReception#getCondition <em>Condition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Condition</em>'.
	 * @see slco.SignalReception#getCondition()
	 * @see #getSignalReception()
	 * @generated
	 */
	EReference getSignalReception_Condition();

	/**
	 * Returns the meta object for class '{@link slco.State <em>State</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>State</em>'.
	 * @see slco.State
	 * @generated
	 */
	EClass getState();

	/**
	 * Returns the meta object for class '{@link slco.StateMachine <em>State Machine</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>State Machine</em>'.
	 * @see slco.StateMachine
	 * @generated
	 */
	EClass getStateMachine();

	/**
	 * Returns the meta object for the attribute '{@link slco.StateMachine#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.StateMachine#getName()
	 * @see #getStateMachine()
	 * @generated
	 */
	EAttribute getStateMachine_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.StateMachine#getVertices <em>Vertices</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Vertices</em>'.
	 * @see slco.StateMachine#getVertices()
	 * @see #getStateMachine()
	 * @generated
	 */
	EReference getStateMachine_Vertices();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.StateMachine#getTransitions <em>Transitions</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Transitions</em>'.
	 * @see slco.StateMachine#getTransitions()
	 * @see #getStateMachine()
	 * @generated
	 */
	EReference getStateMachine_Transitions();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.StateMachine#getVariables <em>Variables</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Variables</em>'.
	 * @see slco.StateMachine#getVariables()
	 * @see #getStateMachine()
	 * @generated
	 */
	EReference getStateMachine_Variables();

	/**
	 * Returns the meta object for class '{@link slco.Statement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Statement</em>'.
	 * @see slco.Statement
	 * @generated
	 */
	EClass getStatement();

	/**
	 * Returns the meta object for class '{@link slco.StringConstantExpression <em>String Constant Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>String Constant Expression</em>'.
	 * @see slco.StringConstantExpression
	 * @generated
	 */
	EClass getStringConstantExpression();

	/**
	 * Returns the meta object for the attribute '{@link slco.StringConstantExpression#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see slco.StringConstantExpression#getValue()
	 * @see #getStringConstantExpression()
	 * @generated
	 */
	EAttribute getStringConstantExpression_Value();

	/**
	 * Returns the meta object for class '{@link slco.TextualStatement <em>Textual Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Textual Statement</em>'.
	 * @see slco.TextualStatement
	 * @generated
	 */
	EClass getTextualStatement();

	/**
	 * Returns the meta object for the attribute '{@link slco.TextualStatement#getStatement <em>Statement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Statement</em>'.
	 * @see slco.TextualStatement#getStatement()
	 * @see #getTextualStatement()
	 * @generated
	 */
	EAttribute getTextualStatement_Statement();

	/**
	 * Returns the meta object for class '{@link slco.Transition <em>Transition</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Transition</em>'.
	 * @see slco.Transition
	 * @generated
	 */
	EClass getTransition();

	/**
	 * Returns the meta object for the attribute '{@link slco.Transition#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Transition#getName()
	 * @see #getTransition()
	 * @generated
	 */
	EAttribute getTransition_Name();

	/**
	 * Returns the meta object for the reference '{@link slco.Transition#getSource <em>Source</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Source</em>'.
	 * @see slco.Transition#getSource()
	 * @see #getTransition()
	 * @generated
	 */
	EReference getTransition_Source();

	/**
	 * Returns the meta object for the reference '{@link slco.Transition#getTarget <em>Target</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Target</em>'.
	 * @see slco.Transition#getTarget()
	 * @see #getTransition()
	 * @generated
	 */
	EReference getTransition_Target();

	/**
	 * Returns the meta object for the containment reference list '{@link slco.Transition#getStatements <em>Statements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Statements</em>'.
	 * @see slco.Transition#getStatements()
	 * @see #getTransition()
	 * @generated
	 */
	EReference getTransition_Statements();

	/**
	 * Returns the meta object for class '{@link slco.UnidirectionalChannel <em>Unidirectional Channel</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Unidirectional Channel</em>'.
	 * @see slco.UnidirectionalChannel
	 * @generated
	 */
	EClass getUnidirectionalChannel();

	/**
	 * Returns the meta object for the reference '{@link slco.UnidirectionalChannel#getSourcePort <em>Source Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Source Port</em>'.
	 * @see slco.UnidirectionalChannel#getSourcePort()
	 * @see #getUnidirectionalChannel()
	 * @generated
	 */
	EReference getUnidirectionalChannel_SourcePort();

	/**
	 * Returns the meta object for the reference '{@link slco.UnidirectionalChannel#getTargetPort <em>Target Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Target Port</em>'.
	 * @see slco.UnidirectionalChannel#getTargetPort()
	 * @see #getUnidirectionalChannel()
	 * @generated
	 */
	EReference getUnidirectionalChannel_TargetPort();

	/**
	 * Returns the meta object for the reference '{@link slco.UnidirectionalChannel#getSourceObject <em>Source Object</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Source Object</em>'.
	 * @see slco.UnidirectionalChannel#getSourceObject()
	 * @see #getUnidirectionalChannel()
	 * @generated
	 */
	EReference getUnidirectionalChannel_SourceObject();

	/**
	 * Returns the meta object for the reference '{@link slco.UnidirectionalChannel#getTargetObject <em>Target Object</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Target Object</em>'.
	 * @see slco.UnidirectionalChannel#getTargetObject()
	 * @see #getUnidirectionalChannel()
	 * @generated
	 */
	EReference getUnidirectionalChannel_TargetObject();

	/**
	 * Returns the meta object for class '{@link slco.Variable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable</em>'.
	 * @see slco.Variable
	 * @generated
	 */
	EClass getVariable();

	/**
	 * Returns the meta object for the attribute '{@link slco.Variable#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Variable#getName()
	 * @see #getVariable()
	 * @generated
	 */
	EAttribute getVariable_Name();

	/**
	 * Returns the meta object for the containment reference '{@link slco.Variable#getInitialValue <em>Initial Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Initial Value</em>'.
	 * @see slco.Variable#getInitialValue()
	 * @see #getVariable()
	 * @generated
	 */
	EReference getVariable_InitialValue();

	/**
	 * Returns the meta object for the attribute '{@link slco.Variable#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see slco.Variable#getType()
	 * @see #getVariable()
	 * @generated
	 */
	EAttribute getVariable_Type();

	/**
	 * Returns the meta object for class '{@link slco.VariableExpression <em>Variable Expression</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Variable Expression</em>'.
	 * @see slco.VariableExpression
	 * @generated
	 */
	EClass getVariableExpression();

	/**
	 * Returns the meta object for the reference '{@link slco.VariableExpression#getVariable <em>Variable</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Variable</em>'.
	 * @see slco.VariableExpression#getVariable()
	 * @see #getVariableExpression()
	 * @generated
	 */
	EReference getVariableExpression_Variable();

	/**
	 * Returns the meta object for class '{@link slco.Vertex <em>Vertex</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Vertex</em>'.
	 * @see slco.Vertex
	 * @generated
	 */
	EClass getVertex();

	/**
	 * Returns the meta object for the attribute '{@link slco.Vertex#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see slco.Vertex#getName()
	 * @see #getVertex()
	 * @generated
	 */
	EAttribute getVertex_Name();

	/**
	 * Returns the meta object for the reference list '{@link slco.Vertex#getIncoming <em>Incoming</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Incoming</em>'.
	 * @see slco.Vertex#getIncoming()
	 * @see #getVertex()
	 * @generated
	 */
	EReference getVertex_Incoming();

	/**
	 * Returns the meta object for the reference list '{@link slco.Vertex#getOutgoing <em>Outgoing</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Outgoing</em>'.
	 * @see slco.Vertex#getOutgoing()
	 * @see #getVertex()
	 * @generated
	 */
	EReference getVertex_Outgoing();

	/**
	 * Returns the meta object for enum '{@link slco.ChannelTypeEnum <em>Channel Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Channel Type Enum</em>'.
	 * @see slco.ChannelTypeEnum
	 * @generated
	 */
	EEnum getChannelTypeEnum();

	/**
	 * Returns the meta object for enum '{@link slco.OperatorEnum <em>Operator Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Operator Enum</em>'.
	 * @see slco.OperatorEnum
	 * @generated
	 */
	EEnum getOperatorEnum();

	/**
	 * Returns the meta object for enum '{@link slco.PrimitiveTypeEnum <em>Primitive Type Enum</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Primitive Type Enum</em>'.
	 * @see slco.PrimitiveTypeEnum
	 * @generated
	 */
	EEnum getPrimitiveTypeEnum();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	SlcoFactory getSlcoFactory();

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
	interface Literals {
		/**
		 * The meta object literal for the '{@link slco.impl.ArgumentTypeImpl <em>Argument Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ArgumentTypeImpl
		 * @see slco.impl.SlcoPackageImpl#getArgumentType()
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
		 * The meta object literal for the '{@link slco.impl.AssignmentImpl <em>Assignment</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.AssignmentImpl
		 * @see slco.impl.SlcoPackageImpl#getAssignment()
		 * @generated
		 */
		EClass ASSIGNMENT = eINSTANCE.getAssignment();

		/**
		 * The meta object literal for the '<em><b>Variable</b></em>' reference feature.
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
		 * The meta object literal for the '{@link slco.impl.BidirectionalChannelImpl <em>Bidirectional Channel</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.BidirectionalChannelImpl
		 * @see slco.impl.SlcoPackageImpl#getBidirectionalChannel()
		 * @generated
		 */
		EClass BIDIRECTIONAL_CHANNEL = eINSTANCE.getBidirectionalChannel();

		/**
		 * The meta object literal for the '<em><b>Port1</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIDIRECTIONAL_CHANNEL__PORT1 = eINSTANCE.getBidirectionalChannel_Port1();

		/**
		 * The meta object literal for the '<em><b>Port2</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIDIRECTIONAL_CHANNEL__PORT2 = eINSTANCE.getBidirectionalChannel_Port2();

		/**
		 * The meta object literal for the '<em><b>Object1</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIDIRECTIONAL_CHANNEL__OBJECT1 = eINSTANCE.getBidirectionalChannel_Object1();

		/**
		 * The meta object literal for the '<em><b>Object2</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BIDIRECTIONAL_CHANNEL__OBJECT2 = eINSTANCE.getBidirectionalChannel_Object2();

		/**
		 * The meta object literal for the '{@link slco.impl.BinaryOperatorExpressionImpl <em>Binary Operator Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.BinaryOperatorExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getBinaryOperatorExpression()
		 * @generated
		 */
		EClass BINARY_OPERATOR_EXPRESSION = eINSTANCE.getBinaryOperatorExpression();

		/**
		 * The meta object literal for the '<em><b>Operator</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute BINARY_OPERATOR_EXPRESSION__OPERATOR = eINSTANCE.getBinaryOperatorExpression_Operator();

		/**
		 * The meta object literal for the '<em><b>Operand1</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BINARY_OPERATOR_EXPRESSION__OPERAND1 = eINSTANCE.getBinaryOperatorExpression_Operand1();

		/**
		 * The meta object literal for the '<em><b>Operand2</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference BINARY_OPERATOR_EXPRESSION__OPERAND2 = eINSTANCE.getBinaryOperatorExpression_Operand2();

		/**
		 * The meta object literal for the '{@link slco.impl.BooleanConstantExpressionImpl <em>Boolean Constant Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.BooleanConstantExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getBooleanConstantExpression()
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
		 * The meta object literal for the '{@link slco.impl.ChannelImpl <em>Channel</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ChannelImpl
		 * @see slco.impl.SlcoPackageImpl#getChannel()
		 * @generated
		 */
		EClass CHANNEL = eINSTANCE.getChannel();

		/**
		 * The meta object literal for the '<em><b>Argument Types</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CHANNEL__ARGUMENT_TYPES = eINSTANCE.getChannel_ArgumentTypes();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CHANNEL__NAME = eINSTANCE.getChannel_Name();

		/**
		 * The meta object literal for the '<em><b>Channel Type</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CHANNEL__CHANNEL_TYPE = eINSTANCE.getChannel_ChannelType();

		/**
		 * The meta object literal for the '{@link slco.impl.ClassImpl <em>Class</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ClassImpl
		 * @see slco.impl.SlcoPackageImpl#getClass_()
		 * @generated
		 */
		EClass CLASS = eINSTANCE.getClass_();

		/**
		 * The meta object literal for the '<em><b>State Machines</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLASS__STATE_MACHINES = eINSTANCE.getClass_StateMachines();

		/**
		 * The meta object literal for the '<em><b>Ports</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLASS__PORTS = eINSTANCE.getClass_Ports();

		/**
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLASS__VARIABLES = eINSTANCE.getClass_Variables();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CLASS__NAME = eINSTANCE.getClass_Name();

		/**
		 * The meta object literal for the '{@link slco.impl.ConstantExpressionImpl <em>Constant Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ConstantExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getConstantExpression()
		 * @generated
		 */
		EClass CONSTANT_EXPRESSION = eINSTANCE.getConstantExpression();

		/**
		 * The meta object literal for the '{@link slco.impl.DelayImpl <em>Delay</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.DelayImpl
		 * @see slco.impl.SlcoPackageImpl#getDelay()
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
		 * The meta object literal for the '{@link slco.impl.ExpressionImpl <em>Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getExpression()
		 * @generated
		 */
		EClass EXPRESSION = eINSTANCE.getExpression();

		/**
		 * The meta object literal for the '{@link slco.impl.FinalImpl <em>Final</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.FinalImpl
		 * @see slco.impl.SlcoPackageImpl#getFinal()
		 * @generated
		 */
		EClass FINAL = eINSTANCE.getFinal();

		/**
		 * The meta object literal for the '{@link slco.impl.InitialImpl <em>Initial</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.InitialImpl
		 * @see slco.impl.SlcoPackageImpl#getInitial()
		 * @generated
		 */
		EClass INITIAL = eINSTANCE.getInitial();

		/**
		 * The meta object literal for the '{@link slco.impl.IntegerConstantExpressionImpl <em>Integer Constant Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.IntegerConstantExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getIntegerConstantExpression()
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
		 * The meta object literal for the '{@link slco.impl.ModelImpl <em>Model</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ModelImpl
		 * @see slco.impl.SlcoPackageImpl#getModel()
		 * @generated
		 */
		EClass MODEL = eINSTANCE.getModel();

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
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MODEL__NAME = eINSTANCE.getModel_Name();

		/**
		 * The meta object literal for the '<em><b>Channels</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MODEL__CHANNELS = eINSTANCE.getModel_Channels();

		/**
		 * The meta object literal for the '{@link slco.impl.ObjectImpl <em>Object</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.ObjectImpl
		 * @see slco.impl.SlcoPackageImpl#getObject()
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
		 * The meta object literal for the '{@link slco.impl.PortImpl <em>Port</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.PortImpl
		 * @see slco.impl.SlcoPackageImpl#getPort()
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
		 * The meta object literal for the '{@link slco.impl.SendSignalImpl <em>Send Signal</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.SendSignalImpl
		 * @see slco.impl.SlcoPackageImpl#getSendSignal()
		 * @generated
		 */
		EClass SEND_SIGNAL = eINSTANCE.getSendSignal();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SEND_SIGNAL__PORT = eINSTANCE.getSendSignal_Port();

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
		 * The meta object literal for the '{@link slco.impl.SignalArgumentImpl <em>Signal Argument</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.SignalArgumentImpl
		 * @see slco.impl.SlcoPackageImpl#getSignalArgument()
		 * @generated
		 */
		EClass SIGNAL_ARGUMENT = eINSTANCE.getSignalArgument();

		/**
		 * The meta object literal for the '{@link slco.impl.SignalArgumentExpressionImpl <em>Signal Argument Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.SignalArgumentExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getSignalArgumentExpression()
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
		 * The meta object literal for the '{@link slco.impl.SignalArgumentVariableImpl <em>Signal Argument Variable</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.SignalArgumentVariableImpl
		 * @see slco.impl.SlcoPackageImpl#getSignalArgumentVariable()
		 * @generated
		 */
		EClass SIGNAL_ARGUMENT_VARIABLE = eINSTANCE.getSignalArgumentVariable();

		/**
		 * The meta object literal for the '<em><b>Variable</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL_ARGUMENT_VARIABLE__VARIABLE = eINSTANCE.getSignalArgumentVariable_Variable();

		/**
		 * The meta object literal for the '{@link slco.impl.SignalReceptionImpl <em>Signal Reception</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.SignalReceptionImpl
		 * @see slco.impl.SlcoPackageImpl#getSignalReception()
		 * @generated
		 */
		EClass SIGNAL_RECEPTION = eINSTANCE.getSignalReception();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL_RECEPTION__PORT = eINSTANCE.getSignalReception_Port();

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
		 * The meta object literal for the '{@link slco.impl.StateImpl <em>State</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.StateImpl
		 * @see slco.impl.SlcoPackageImpl#getState()
		 * @generated
		 */
		EClass STATE = eINSTANCE.getState();

		/**
		 * The meta object literal for the '{@link slco.impl.StateMachineImpl <em>State Machine</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.StateMachineImpl
		 * @see slco.impl.SlcoPackageImpl#getStateMachine()
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
		 * The meta object literal for the '<em><b>Variables</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference STATE_MACHINE__VARIABLES = eINSTANCE.getStateMachine_Variables();

		/**
		 * The meta object literal for the '{@link slco.impl.StatementImpl <em>Statement</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.StatementImpl
		 * @see slco.impl.SlcoPackageImpl#getStatement()
		 * @generated
		 */
		EClass STATEMENT = eINSTANCE.getStatement();

		/**
		 * The meta object literal for the '{@link slco.impl.StringConstantExpressionImpl <em>String Constant Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.StringConstantExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getStringConstantExpression()
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
		 * The meta object literal for the '{@link slco.impl.TextualStatementImpl <em>Textual Statement</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.TextualStatementImpl
		 * @see slco.impl.SlcoPackageImpl#getTextualStatement()
		 * @generated
		 */
		EClass TEXTUAL_STATEMENT = eINSTANCE.getTextualStatement();

		/**
		 * The meta object literal for the '<em><b>Statement</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute TEXTUAL_STATEMENT__STATEMENT = eINSTANCE.getTextualStatement_Statement();

		/**
		 * The meta object literal for the '{@link slco.impl.TransitionImpl <em>Transition</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.TransitionImpl
		 * @see slco.impl.SlcoPackageImpl#getTransition()
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
		 * The meta object literal for the '<em><b>Source</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TRANSITION__SOURCE = eINSTANCE.getTransition_Source();

		/**
		 * The meta object literal for the '<em><b>Target</b></em>' reference feature.
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
		 * The meta object literal for the '{@link slco.impl.UnidirectionalChannelImpl <em>Unidirectional Channel</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.UnidirectionalChannelImpl
		 * @see slco.impl.SlcoPackageImpl#getUnidirectionalChannel()
		 * @generated
		 */
		EClass UNIDIRECTIONAL_CHANNEL = eINSTANCE.getUnidirectionalChannel();

		/**
		 * The meta object literal for the '<em><b>Source Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UNIDIRECTIONAL_CHANNEL__SOURCE_PORT = eINSTANCE.getUnidirectionalChannel_SourcePort();

		/**
		 * The meta object literal for the '<em><b>Target Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UNIDIRECTIONAL_CHANNEL__TARGET_PORT = eINSTANCE.getUnidirectionalChannel_TargetPort();

		/**
		 * The meta object literal for the '<em><b>Source Object</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UNIDIRECTIONAL_CHANNEL__SOURCE_OBJECT = eINSTANCE.getUnidirectionalChannel_SourceObject();

		/**
		 * The meta object literal for the '<em><b>Target Object</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference UNIDIRECTIONAL_CHANNEL__TARGET_OBJECT = eINSTANCE.getUnidirectionalChannel_TargetObject();

		/**
		 * The meta object literal for the '{@link slco.impl.VariableImpl <em>Variable</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.VariableImpl
		 * @see slco.impl.SlcoPackageImpl#getVariable()
		 * @generated
		 */
		EClass VARIABLE = eINSTANCE.getVariable();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute VARIABLE__NAME = eINSTANCE.getVariable_Name();

		/**
		 * The meta object literal for the '<em><b>Initial Value</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARIABLE__INITIAL_VALUE = eINSTANCE.getVariable_InitialValue();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute VARIABLE__TYPE = eINSTANCE.getVariable_Type();

		/**
		 * The meta object literal for the '{@link slco.impl.VariableExpressionImpl <em>Variable Expression</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.VariableExpressionImpl
		 * @see slco.impl.SlcoPackageImpl#getVariableExpression()
		 * @generated
		 */
		EClass VARIABLE_EXPRESSION = eINSTANCE.getVariableExpression();

		/**
		 * The meta object literal for the '<em><b>Variable</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VARIABLE_EXPRESSION__VARIABLE = eINSTANCE.getVariableExpression_Variable();

		/**
		 * The meta object literal for the '{@link slco.impl.VertexImpl <em>Vertex</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.impl.VertexImpl
		 * @see slco.impl.SlcoPackageImpl#getVertex()
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
		 * The meta object literal for the '<em><b>Incoming</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VERTEX__INCOMING = eINSTANCE.getVertex_Incoming();

		/**
		 * The meta object literal for the '<em><b>Outgoing</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference VERTEX__OUTGOING = eINSTANCE.getVertex_Outgoing();

		/**
		 * The meta object literal for the '{@link slco.ChannelTypeEnum <em>Channel Type Enum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.ChannelTypeEnum
		 * @see slco.impl.SlcoPackageImpl#getChannelTypeEnum()
		 * @generated
		 */
		EEnum CHANNEL_TYPE_ENUM = eINSTANCE.getChannelTypeEnum();

		/**
		 * The meta object literal for the '{@link slco.OperatorEnum <em>Operator Enum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.OperatorEnum
		 * @see slco.impl.SlcoPackageImpl#getOperatorEnum()
		 * @generated
		 */
		EEnum OPERATOR_ENUM = eINSTANCE.getOperatorEnum();

		/**
		 * The meta object literal for the '{@link slco.PrimitiveTypeEnum <em>Primitive Type Enum</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco.PrimitiveTypeEnum
		 * @see slco.impl.SlcoPackageImpl#getPrimitiveTypeEnum()
		 * @generated
		 */
		EEnum PRIMITIVE_TYPE_ENUM = eINSTANCE.getPrimitiveTypeEnum();

	}

} //SlcoPackage

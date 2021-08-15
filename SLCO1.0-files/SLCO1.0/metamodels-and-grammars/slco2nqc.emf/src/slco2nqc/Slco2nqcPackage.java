/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
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
 * @see slco2nqc.Slco2nqcFactory
 * @model kind="package"
 * @generated
 */
public interface Slco2nqcPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "slco2nqc";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "slco2nqc";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "slco2nqc";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	Slco2nqcPackage eINSTANCE = slco2nqc.impl.Slco2nqcPackageImpl.init();

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Signal2MotorListImpl <em>Signal2 Motor List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Signal2MotorListImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2MotorList()
	 * @generated
	 */
	int SIGNAL2_MOTOR_LIST = 0;

	/**
	 * The feature id for the '<em><b>Signal2 Motors</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS = 0;

	/**
	 * The number of structural features of the '<em>Signal2 Motor List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Signal2SensorListImpl <em>Signal2 Sensor List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Signal2SensorListImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2SensorList()
	 * @generated
	 */
	int SIGNAL2_SENSOR_LIST = 1;

	/**
	 * The feature id for the '<em><b>Signal2 Sensors</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS = 0;

	/**
	 * The number of structural features of the '<em>Signal2 Sensor List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.MotorInitListImpl <em>Motor Init List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.MotorInitListImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getMotorInitList()
	 * @generated
	 */
	int MOTOR_INIT_LIST = 2;

	/**
	 * The feature id for the '<em><b>Motor Inits</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT_LIST__MOTOR_INITS = 0;

	/**
	 * The number of structural features of the '<em>Motor Init List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.MotorInitImpl <em>Motor Init</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.MotorInitImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getMotorInit()
	 * @generated
	 */
	int MOTOR_INIT = 3;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT__PORT = 0;

	/**
	 * The feature id for the '<em><b>Power</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT__POWER = 1;

	/**
	 * The feature id for the '<em><b>Output Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT__OUTPUT_NAME = 2;

	/**
	 * The number of structural features of the '<em>Motor Init</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOTOR_INIT_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.ClassListImpl <em>Class List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.ClassListImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getClassList()
	 * @generated
	 */
	int CLASS_LIST = 4;

	/**
	 * The feature id for the '<em><b>Classes</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS_LIST__CLASSES = 0;

	/**
	 * The number of structural features of the '<em>Class List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLASS_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Signal2MotorImpl <em>Signal2 Motor</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Signal2MotorImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2Motor()
	 * @generated
	 */
	int SIGNAL2_MOTOR = 5;

	/**
	 * The feature id for the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR__SIGNAL_NAME = 0;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR__PORT = 1;

	/**
	 * The feature id for the '<em><b>Output Mode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR__OUTPUT_MODE = 2;

	/**
	 * The number of structural features of the '<em>Signal2 Motor</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_FEATURE_COUNT = 3;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Signal2MotorWithDirectionImpl <em>Signal2 Motor With Direction</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Signal2MotorWithDirectionImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2MotorWithDirection()
	 * @generated
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION = 6;

	/**
	 * The feature id for the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION__SIGNAL_NAME = SIGNAL2_MOTOR__SIGNAL_NAME;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION__PORT = SIGNAL2_MOTOR__PORT;

	/**
	 * The feature id for the '<em><b>Output Mode</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_MODE = SIGNAL2_MOTOR__OUTPUT_MODE;

	/**
	 * The feature id for the '<em><b>Output Direction</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION = SIGNAL2_MOTOR_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Signal2 Motor With Direction</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_MOTOR_WITH_DIRECTION_FEATURE_COUNT = SIGNAL2_MOTOR_FEATURE_COUNT + 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Signal2SensorImpl <em>Signal2 Sensor</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Signal2SensorImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2Sensor()
	 * @generated
	 */
	int SIGNAL2_SENSOR = 7;

	/**
	 * The feature id for the '<em><b>Signal Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR__SIGNAL_NAME = 0;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR__PORT = 1;

	/**
	 * The feature id for the '<em><b>Sensor Is High</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR__SENSOR_IS_HIGH = 2;

	/**
	 * The feature id for the '<em><b>Treshold</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR__TRESHOLD = 3;

	/**
	 * The number of structural features of the '<em>Signal2 Sensor</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SIGNAL2_SENSOR_FEATURE_COUNT = 4;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.Slco2NqcImpl <em>Slco2 Nqc</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.Slco2NqcImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSlco2Nqc()
	 * @generated
	 */
	int SLCO2_NQC = 8;

	/**
	 * The feature id for the '<em><b>Signal2 Motor List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC__SIGNAL2_MOTOR_LIST = 0;

	/**
	 * The feature id for the '<em><b>Signal2 Sensor List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC__SIGNAL2_SENSOR_LIST = 1;

	/**
	 * The feature id for the '<em><b>Motor Init List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC__MOTOR_INIT_LIST = 2;

	/**
	 * The feature id for the '<em><b>Class List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC__CLASS_LIST = 3;

	/**
	 * The feature id for the '<em><b>Sensor Init List</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC__SENSOR_INIT_LIST = 4;

	/**
	 * The number of structural features of the '<em>Slco2 Nqc</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SLCO2_NQC_FEATURE_COUNT = 5;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.SensorInitListImpl <em>Sensor Init List</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.SensorInitListImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSensorInitList()
	 * @generated
	 */
	int SENSOR_INIT_LIST = 9;

	/**
	 * The feature id for the '<em><b>Sensor Inits</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT_LIST__SENSOR_INITS = 0;

	/**
	 * The number of structural features of the '<em>Sensor Init List</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT_LIST_FEATURE_COUNT = 1;

	/**
	 * The meta object id for the '{@link slco2nqc.impl.SensorInitImpl <em>Sensor Init</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see slco2nqc.impl.SensorInitImpl
	 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSensorInit()
	 * @generated
	 */
	int SENSOR_INIT = 10;

	/**
	 * The feature id for the '<em><b>Sensor Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT__SENSOR_NAME = 0;

	/**
	 * The feature id for the '<em><b>Sensor Configuration</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT__SENSOR_CONFIGURATION = 1;

	/**
	 * The feature id for the '<em><b>Port</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT__PORT = 2;

	/**
	 * The number of structural features of the '<em>Sensor Init</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SENSOR_INIT_FEATURE_COUNT = 3;


	/**
	 * Returns the meta object for class '{@link slco2nqc.Signal2MotorList <em>Signal2 Motor List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal2 Motor List</em>'.
	 * @see slco2nqc.Signal2MotorList
	 * @generated
	 */
	EClass getSignal2MotorList();

	/**
	 * Returns the meta object for the containment reference list '{@link slco2nqc.Signal2MotorList#getSignal2Motors <em>Signal2 Motors</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Signal2 Motors</em>'.
	 * @see slco2nqc.Signal2MotorList#getSignal2Motors()
	 * @see #getSignal2MotorList()
	 * @generated
	 */
	EReference getSignal2MotorList_Signal2Motors();

	/**
	 * Returns the meta object for class '{@link slco2nqc.Signal2SensorList <em>Signal2 Sensor List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal2 Sensor List</em>'.
	 * @see slco2nqc.Signal2SensorList
	 * @generated
	 */
	EClass getSignal2SensorList();

	/**
	 * Returns the meta object for the containment reference list '{@link slco2nqc.Signal2SensorList#getSignal2Sensors <em>Signal2 Sensors</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Signal2 Sensors</em>'.
	 * @see slco2nqc.Signal2SensorList#getSignal2Sensors()
	 * @see #getSignal2SensorList()
	 * @generated
	 */
	EReference getSignal2SensorList_Signal2Sensors();

	/**
	 * Returns the meta object for class '{@link slco2nqc.MotorInitList <em>Motor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Motor Init List</em>'.
	 * @see slco2nqc.MotorInitList
	 * @generated
	 */
	EClass getMotorInitList();

	/**
	 * Returns the meta object for the containment reference list '{@link slco2nqc.MotorInitList#getMotorInits <em>Motor Inits</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Motor Inits</em>'.
	 * @see slco2nqc.MotorInitList#getMotorInits()
	 * @see #getMotorInitList()
	 * @generated
	 */
	EReference getMotorInitList_MotorInits();

	/**
	 * Returns the meta object for class '{@link slco2nqc.MotorInit <em>Motor Init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Motor Init</em>'.
	 * @see slco2nqc.MotorInit
	 * @generated
	 */
	EClass getMotorInit();

	/**
	 * Returns the meta object for the reference '{@link slco2nqc.MotorInit#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco2nqc.MotorInit#getPort()
	 * @see #getMotorInit()
	 * @generated
	 */
	EReference getMotorInit_Port();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.MotorInit#getPower <em>Power</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Power</em>'.
	 * @see slco2nqc.MotorInit#getPower()
	 * @see #getMotorInit()
	 * @generated
	 */
	EAttribute getMotorInit_Power();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.MotorInit#getOutputName <em>Output Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Output Name</em>'.
	 * @see slco2nqc.MotorInit#getOutputName()
	 * @see #getMotorInit()
	 * @generated
	 */
	EAttribute getMotorInit_OutputName();

	/**
	 * Returns the meta object for class '{@link slco2nqc.ClassList <em>Class List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Class List</em>'.
	 * @see slco2nqc.ClassList
	 * @generated
	 */
	EClass getClassList();

	/**
	 * Returns the meta object for the reference list '{@link slco2nqc.ClassList#getClasses <em>Classes</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Classes</em>'.
	 * @see slco2nqc.ClassList#getClasses()
	 * @see #getClassList()
	 * @generated
	 */
	EReference getClassList_Classes();

	/**
	 * Returns the meta object for class '{@link slco2nqc.Signal2Motor <em>Signal2 Motor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal2 Motor</em>'.
	 * @see slco2nqc.Signal2Motor
	 * @generated
	 */
	EClass getSignal2Motor();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2Motor#getSignalName <em>Signal Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Signal Name</em>'.
	 * @see slco2nqc.Signal2Motor#getSignalName()
	 * @see #getSignal2Motor()
	 * @generated
	 */
	EAttribute getSignal2Motor_SignalName();

	/**
	 * Returns the meta object for the reference '{@link slco2nqc.Signal2Motor#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco2nqc.Signal2Motor#getPort()
	 * @see #getSignal2Motor()
	 * @generated
	 */
	EReference getSignal2Motor_Port();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2Motor#getOutputMode <em>Output Mode</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Output Mode</em>'.
	 * @see slco2nqc.Signal2Motor#getOutputMode()
	 * @see #getSignal2Motor()
	 * @generated
	 */
	EAttribute getSignal2Motor_OutputMode();

	/**
	 * Returns the meta object for class '{@link slco2nqc.Signal2MotorWithDirection <em>Signal2 Motor With Direction</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal2 Motor With Direction</em>'.
	 * @see slco2nqc.Signal2MotorWithDirection
	 * @generated
	 */
	EClass getSignal2MotorWithDirection();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2MotorWithDirection#getOutputDirection <em>Output Direction</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Output Direction</em>'.
	 * @see slco2nqc.Signal2MotorWithDirection#getOutputDirection()
	 * @see #getSignal2MotorWithDirection()
	 * @generated
	 */
	EAttribute getSignal2MotorWithDirection_OutputDirection();

	/**
	 * Returns the meta object for class '{@link slco2nqc.Signal2Sensor <em>Signal2 Sensor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Signal2 Sensor</em>'.
	 * @see slco2nqc.Signal2Sensor
	 * @generated
	 */
	EClass getSignal2Sensor();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2Sensor#getSignalName <em>Signal Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Signal Name</em>'.
	 * @see slco2nqc.Signal2Sensor#getSignalName()
	 * @see #getSignal2Sensor()
	 * @generated
	 */
	EAttribute getSignal2Sensor_SignalName();

	/**
	 * Returns the meta object for the reference '{@link slco2nqc.Signal2Sensor#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco2nqc.Signal2Sensor#getPort()
	 * @see #getSignal2Sensor()
	 * @generated
	 */
	EReference getSignal2Sensor_Port();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2Sensor#isSensorIsHigh <em>Sensor Is High</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Is High</em>'.
	 * @see slco2nqc.Signal2Sensor#isSensorIsHigh()
	 * @see #getSignal2Sensor()
	 * @generated
	 */
	EAttribute getSignal2Sensor_SensorIsHigh();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.Signal2Sensor#getTreshold <em>Treshold</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Treshold</em>'.
	 * @see slco2nqc.Signal2Sensor#getTreshold()
	 * @see #getSignal2Sensor()
	 * @generated
	 */
	EAttribute getSignal2Sensor_Treshold();

	/**
	 * Returns the meta object for class '{@link slco2nqc.Slco2Nqc <em>Slco2 Nqc</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Slco2 Nqc</em>'.
	 * @see slco2nqc.Slco2Nqc
	 * @generated
	 */
	EClass getSlco2Nqc();

	/**
	 * Returns the meta object for the containment reference '{@link slco2nqc.Slco2Nqc#getSignal2MotorList <em>Signal2 Motor List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Signal2 Motor List</em>'.
	 * @see slco2nqc.Slco2Nqc#getSignal2MotorList()
	 * @see #getSlco2Nqc()
	 * @generated
	 */
	EReference getSlco2Nqc_Signal2MotorList();

	/**
	 * Returns the meta object for the containment reference '{@link slco2nqc.Slco2Nqc#getSignal2SensorList <em>Signal2 Sensor List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Signal2 Sensor List</em>'.
	 * @see slco2nqc.Slco2Nqc#getSignal2SensorList()
	 * @see #getSlco2Nqc()
	 * @generated
	 */
	EReference getSlco2Nqc_Signal2SensorList();

	/**
	 * Returns the meta object for the containment reference '{@link slco2nqc.Slco2Nqc#getMotorInitList <em>Motor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Motor Init List</em>'.
	 * @see slco2nqc.Slco2Nqc#getMotorInitList()
	 * @see #getSlco2Nqc()
	 * @generated
	 */
	EReference getSlco2Nqc_MotorInitList();

	/**
	 * Returns the meta object for the containment reference '{@link slco2nqc.Slco2Nqc#getClassList <em>Class List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Class List</em>'.
	 * @see slco2nqc.Slco2Nqc#getClassList()
	 * @see #getSlco2Nqc()
	 * @generated
	 */
	EReference getSlco2Nqc_ClassList();

	/**
	 * Returns the meta object for the containment reference '{@link slco2nqc.Slco2Nqc#getSensorInitList <em>Sensor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Sensor Init List</em>'.
	 * @see slco2nqc.Slco2Nqc#getSensorInitList()
	 * @see #getSlco2Nqc()
	 * @generated
	 */
	EReference getSlco2Nqc_SensorInitList();

	/**
	 * Returns the meta object for class '{@link slco2nqc.SensorInitList <em>Sensor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Init List</em>'.
	 * @see slco2nqc.SensorInitList
	 * @generated
	 */
	EClass getSensorInitList();

	/**
	 * Returns the meta object for the containment reference list '{@link slco2nqc.SensorInitList#getSensorInits <em>Sensor Inits</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Sensor Inits</em>'.
	 * @see slco2nqc.SensorInitList#getSensorInits()
	 * @see #getSensorInitList()
	 * @generated
	 */
	EReference getSensorInitList_SensorInits();

	/**
	 * Returns the meta object for class '{@link slco2nqc.SensorInit <em>Sensor Init</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Sensor Init</em>'.
	 * @see slco2nqc.SensorInit
	 * @generated
	 */
	EClass getSensorInit();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.SensorInit#getSensorName <em>Sensor Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Name</em>'.
	 * @see slco2nqc.SensorInit#getSensorName()
	 * @see #getSensorInit()
	 * @generated
	 */
	EAttribute getSensorInit_SensorName();

	/**
	 * Returns the meta object for the attribute '{@link slco2nqc.SensorInit#getSensorConfiguration <em>Sensor Configuration</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sensor Configuration</em>'.
	 * @see slco2nqc.SensorInit#getSensorConfiguration()
	 * @see #getSensorInit()
	 * @generated
	 */
	EAttribute getSensorInit_SensorConfiguration();

	/**
	 * Returns the meta object for the reference '{@link slco2nqc.SensorInit#getPort <em>Port</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Port</em>'.
	 * @see slco2nqc.SensorInit#getPort()
	 * @see #getSensorInit()
	 * @generated
	 */
	EReference getSensorInit_Port();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	Slco2nqcFactory getSlco2nqcFactory();

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
		 * The meta object literal for the '{@link slco2nqc.impl.Signal2MotorListImpl <em>Signal2 Motor List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Signal2MotorListImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2MotorList()
		 * @generated
		 */
		EClass SIGNAL2_MOTOR_LIST = eINSTANCE.getSignal2MotorList();

		/**
		 * The meta object literal for the '<em><b>Signal2 Motors</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS = eINSTANCE.getSignal2MotorList_Signal2Motors();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.Signal2SensorListImpl <em>Signal2 Sensor List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Signal2SensorListImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2SensorList()
		 * @generated
		 */
		EClass SIGNAL2_SENSOR_LIST = eINSTANCE.getSignal2SensorList();

		/**
		 * The meta object literal for the '<em><b>Signal2 Sensors</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS = eINSTANCE.getSignal2SensorList_Signal2Sensors();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.MotorInitListImpl <em>Motor Init List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.MotorInitListImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getMotorInitList()
		 * @generated
		 */
		EClass MOTOR_INIT_LIST = eINSTANCE.getMotorInitList();

		/**
		 * The meta object literal for the '<em><b>Motor Inits</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MOTOR_INIT_LIST__MOTOR_INITS = eINSTANCE.getMotorInitList_MotorInits();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.MotorInitImpl <em>Motor Init</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.MotorInitImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getMotorInit()
		 * @generated
		 */
		EClass MOTOR_INIT = eINSTANCE.getMotorInit();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MOTOR_INIT__PORT = eINSTANCE.getMotorInit_Port();

		/**
		 * The meta object literal for the '<em><b>Power</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MOTOR_INIT__POWER = eINSTANCE.getMotorInit_Power();

		/**
		 * The meta object literal for the '<em><b>Output Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MOTOR_INIT__OUTPUT_NAME = eINSTANCE.getMotorInit_OutputName();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.ClassListImpl <em>Class List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.ClassListImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getClassList()
		 * @generated
		 */
		EClass CLASS_LIST = eINSTANCE.getClassList();

		/**
		 * The meta object literal for the '<em><b>Classes</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLASS_LIST__CLASSES = eINSTANCE.getClassList_Classes();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.Signal2MotorImpl <em>Signal2 Motor</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Signal2MotorImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2Motor()
		 * @generated
		 */
		EClass SIGNAL2_MOTOR = eINSTANCE.getSignal2Motor();

		/**
		 * The meta object literal for the '<em><b>Signal Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_MOTOR__SIGNAL_NAME = eINSTANCE.getSignal2Motor_SignalName();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL2_MOTOR__PORT = eINSTANCE.getSignal2Motor_Port();

		/**
		 * The meta object literal for the '<em><b>Output Mode</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_MOTOR__OUTPUT_MODE = eINSTANCE.getSignal2Motor_OutputMode();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.Signal2MotorWithDirectionImpl <em>Signal2 Motor With Direction</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Signal2MotorWithDirectionImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2MotorWithDirection()
		 * @generated
		 */
		EClass SIGNAL2_MOTOR_WITH_DIRECTION = eINSTANCE.getSignal2MotorWithDirection();

		/**
		 * The meta object literal for the '<em><b>Output Direction</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION = eINSTANCE.getSignal2MotorWithDirection_OutputDirection();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.Signal2SensorImpl <em>Signal2 Sensor</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Signal2SensorImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSignal2Sensor()
		 * @generated
		 */
		EClass SIGNAL2_SENSOR = eINSTANCE.getSignal2Sensor();

		/**
		 * The meta object literal for the '<em><b>Signal Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_SENSOR__SIGNAL_NAME = eINSTANCE.getSignal2Sensor_SignalName();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SIGNAL2_SENSOR__PORT = eINSTANCE.getSignal2Sensor_Port();

		/**
		 * The meta object literal for the '<em><b>Sensor Is High</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_SENSOR__SENSOR_IS_HIGH = eINSTANCE.getSignal2Sensor_SensorIsHigh();

		/**
		 * The meta object literal for the '<em><b>Treshold</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SIGNAL2_SENSOR__TRESHOLD = eINSTANCE.getSignal2Sensor_Treshold();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.Slco2NqcImpl <em>Slco2 Nqc</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.Slco2NqcImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSlco2Nqc()
		 * @generated
		 */
		EClass SLCO2_NQC = eINSTANCE.getSlco2Nqc();

		/**
		 * The meta object literal for the '<em><b>Signal2 Motor List</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SLCO2_NQC__SIGNAL2_MOTOR_LIST = eINSTANCE.getSlco2Nqc_Signal2MotorList();

		/**
		 * The meta object literal for the '<em><b>Signal2 Sensor List</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SLCO2_NQC__SIGNAL2_SENSOR_LIST = eINSTANCE.getSlco2Nqc_Signal2SensorList();

		/**
		 * The meta object literal for the '<em><b>Motor Init List</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SLCO2_NQC__MOTOR_INIT_LIST = eINSTANCE.getSlco2Nqc_MotorInitList();

		/**
		 * The meta object literal for the '<em><b>Class List</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SLCO2_NQC__CLASS_LIST = eINSTANCE.getSlco2Nqc_ClassList();

		/**
		 * The meta object literal for the '<em><b>Sensor Init List</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SLCO2_NQC__SENSOR_INIT_LIST = eINSTANCE.getSlco2Nqc_SensorInitList();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.SensorInitListImpl <em>Sensor Init List</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.SensorInitListImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSensorInitList()
		 * @generated
		 */
		EClass SENSOR_INIT_LIST = eINSTANCE.getSensorInitList();

		/**
		 * The meta object literal for the '<em><b>Sensor Inits</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SENSOR_INIT_LIST__SENSOR_INITS = eINSTANCE.getSensorInitList_SensorInits();

		/**
		 * The meta object literal for the '{@link slco2nqc.impl.SensorInitImpl <em>Sensor Init</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see slco2nqc.impl.SensorInitImpl
		 * @see slco2nqc.impl.Slco2nqcPackageImpl#getSensorInit()
		 * @generated
		 */
		EClass SENSOR_INIT = eINSTANCE.getSensorInit();

		/**
		 * The meta object literal for the '<em><b>Sensor Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SENSOR_INIT__SENSOR_NAME = eINSTANCE.getSensorInit_SensorName();

		/**
		 * The meta object literal for the '<em><b>Sensor Configuration</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SENSOR_INIT__SENSOR_CONFIGURATION = eINSTANCE.getSensorInit_SensorConfiguration();

		/**
		 * The meta object literal for the '<em><b>Port</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SENSOR_INIT__PORT = eINSTANCE.getSensorInit_Port();

	}

} //Slco2nqcPackage

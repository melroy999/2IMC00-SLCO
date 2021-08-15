/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import nqc.NqcPackage;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

import org.eclipse.emf.ecore.impl.EPackageImpl;

import slco.SlcoPackage;

import slco2nqc.ClassList;
import slco2nqc.MotorInit;
import slco2nqc.MotorInitList;
import slco2nqc.SensorInit;
import slco2nqc.SensorInitList;
import slco2nqc.Signal2Motor;
import slco2nqc.Signal2MotorList;
import slco2nqc.Signal2MotorWithDirection;
import slco2nqc.Signal2Sensor;
import slco2nqc.Signal2SensorList;
import slco2nqc.Slco2Nqc;
import slco2nqc.Slco2nqcFactory;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class Slco2nqcPackageImpl extends EPackageImpl implements Slco2nqcPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signal2MotorListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signal2SensorListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass motorInitListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass motorInitEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass classListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signal2MotorEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signal2MotorWithDirectionEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass signal2SensorEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass slco2NqcEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorInitListEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass sensorInitEClass = null;

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
	 * @see slco2nqc.Slco2nqcPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private Slco2nqcPackageImpl() {
		super(eNS_URI, Slco2nqcFactory.eINSTANCE);
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
	 * <p>This method is used to initialize {@link Slco2nqcPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static Slco2nqcPackage init() {
		if (isInited) return (Slco2nqcPackage)EPackage.Registry.INSTANCE.getEPackage(Slco2nqcPackage.eNS_URI);

		// Obtain or create and register package
		Slco2nqcPackageImpl theSlco2nqcPackage = (Slco2nqcPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof Slco2nqcPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new Slco2nqcPackageImpl());

		isInited = true;

		// Initialize simple dependencies
		NqcPackage.eINSTANCE.eClass();
		SlcoPackage.eINSTANCE.eClass();

		// Create package meta-data objects
		theSlco2nqcPackage.createPackageContents();

		// Initialize created meta-data
		theSlco2nqcPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theSlco2nqcPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(Slco2nqcPackage.eNS_URI, theSlco2nqcPackage);
		return theSlco2nqcPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignal2MotorList() {
		return signal2MotorListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignal2MotorList_Signal2Motors() {
		return (EReference)signal2MotorListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignal2SensorList() {
		return signal2SensorListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignal2SensorList_Signal2Sensors() {
		return (EReference)signal2SensorListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMotorInitList() {
		return motorInitListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMotorInitList_MotorInits() {
		return (EReference)motorInitListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMotorInit() {
		return motorInitEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMotorInit_Port() {
		return (EReference)motorInitEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMotorInit_Power() {
		return (EAttribute)motorInitEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMotorInit_OutputName() {
		return (EAttribute)motorInitEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getClassList() {
		return classListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getClassList_Classes() {
		return (EReference)classListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignal2Motor() {
		return signal2MotorEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2Motor_SignalName() {
		return (EAttribute)signal2MotorEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignal2Motor_Port() {
		return (EReference)signal2MotorEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2Motor_OutputMode() {
		return (EAttribute)signal2MotorEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignal2MotorWithDirection() {
		return signal2MotorWithDirectionEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2MotorWithDirection_OutputDirection() {
		return (EAttribute)signal2MotorWithDirectionEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSignal2Sensor() {
		return signal2SensorEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2Sensor_SignalName() {
		return (EAttribute)signal2SensorEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSignal2Sensor_Port() {
		return (EReference)signal2SensorEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2Sensor_SensorIsHigh() {
		return (EAttribute)signal2SensorEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSignal2Sensor_Treshold() {
		return (EAttribute)signal2SensorEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSlco2Nqc() {
		return slco2NqcEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSlco2Nqc_Signal2MotorList() {
		return (EReference)slco2NqcEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSlco2Nqc_Signal2SensorList() {
		return (EReference)slco2NqcEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSlco2Nqc_MotorInitList() {
		return (EReference)slco2NqcEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSlco2Nqc_ClassList() {
		return (EReference)slco2NqcEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSlco2Nqc_SensorInitList() {
		return (EReference)slco2NqcEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorInitList() {
		return sensorInitListEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSensorInitList_SensorInits() {
		return (EReference)sensorInitListEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getSensorInit() {
		return sensorInitEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorInit_SensorName() {
		return (EAttribute)sensorInitEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getSensorInit_SensorConfiguration() {
		return (EAttribute)sensorInitEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getSensorInit_Port() {
		return (EReference)sensorInitEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2nqcFactory getSlco2nqcFactory() {
		return (Slco2nqcFactory)getEFactoryInstance();
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
	public void createPackageContents() {
		if (isCreated) return;
		isCreated = true;

		// Create classes and their features
		signal2MotorListEClass = createEClass(SIGNAL2_MOTOR_LIST);
		createEReference(signal2MotorListEClass, SIGNAL2_MOTOR_LIST__SIGNAL2_MOTORS);

		signal2SensorListEClass = createEClass(SIGNAL2_SENSOR_LIST);
		createEReference(signal2SensorListEClass, SIGNAL2_SENSOR_LIST__SIGNAL2_SENSORS);

		motorInitListEClass = createEClass(MOTOR_INIT_LIST);
		createEReference(motorInitListEClass, MOTOR_INIT_LIST__MOTOR_INITS);

		motorInitEClass = createEClass(MOTOR_INIT);
		createEReference(motorInitEClass, MOTOR_INIT__PORT);
		createEAttribute(motorInitEClass, MOTOR_INIT__POWER);
		createEAttribute(motorInitEClass, MOTOR_INIT__OUTPUT_NAME);

		classListEClass = createEClass(CLASS_LIST);
		createEReference(classListEClass, CLASS_LIST__CLASSES);

		signal2MotorEClass = createEClass(SIGNAL2_MOTOR);
		createEAttribute(signal2MotorEClass, SIGNAL2_MOTOR__SIGNAL_NAME);
		createEReference(signal2MotorEClass, SIGNAL2_MOTOR__PORT);
		createEAttribute(signal2MotorEClass, SIGNAL2_MOTOR__OUTPUT_MODE);

		signal2MotorWithDirectionEClass = createEClass(SIGNAL2_MOTOR_WITH_DIRECTION);
		createEAttribute(signal2MotorWithDirectionEClass, SIGNAL2_MOTOR_WITH_DIRECTION__OUTPUT_DIRECTION);

		signal2SensorEClass = createEClass(SIGNAL2_SENSOR);
		createEAttribute(signal2SensorEClass, SIGNAL2_SENSOR__SIGNAL_NAME);
		createEReference(signal2SensorEClass, SIGNAL2_SENSOR__PORT);
		createEAttribute(signal2SensorEClass, SIGNAL2_SENSOR__SENSOR_IS_HIGH);
		createEAttribute(signal2SensorEClass, SIGNAL2_SENSOR__TRESHOLD);

		slco2NqcEClass = createEClass(SLCO2_NQC);
		createEReference(slco2NqcEClass, SLCO2_NQC__SIGNAL2_MOTOR_LIST);
		createEReference(slco2NqcEClass, SLCO2_NQC__SIGNAL2_SENSOR_LIST);
		createEReference(slco2NqcEClass, SLCO2_NQC__MOTOR_INIT_LIST);
		createEReference(slco2NqcEClass, SLCO2_NQC__CLASS_LIST);
		createEReference(slco2NqcEClass, SLCO2_NQC__SENSOR_INIT_LIST);

		sensorInitListEClass = createEClass(SENSOR_INIT_LIST);
		createEReference(sensorInitListEClass, SENSOR_INIT_LIST__SENSOR_INITS);

		sensorInitEClass = createEClass(SENSOR_INIT);
		createEAttribute(sensorInitEClass, SENSOR_INIT__SENSOR_NAME);
		createEAttribute(sensorInitEClass, SENSOR_INIT__SENSOR_CONFIGURATION);
		createEReference(sensorInitEClass, SENSOR_INIT__PORT);
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
	public void initializePackageContents() {
		if (isInitialized) return;
		isInitialized = true;

		// Initialize package
		setName(eNAME);
		setNsPrefix(eNS_PREFIX);
		setNsURI(eNS_URI);

		// Obtain other dependent packages
		SlcoPackage theSlcoPackage = (SlcoPackage)EPackage.Registry.INSTANCE.getEPackage(SlcoPackage.eNS_URI);
		NqcPackage theNqcPackage = (NqcPackage)EPackage.Registry.INSTANCE.getEPackage(NqcPackage.eNS_URI);

		// Create type parameters

		// Set bounds for type parameters

		// Add supertypes to classes
		signal2MotorWithDirectionEClass.getESuperTypes().add(this.getSignal2Motor());

		// Initialize classes and features; add operations and parameters
		initEClass(signal2MotorListEClass, Signal2MotorList.class, "Signal2MotorList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSignal2MotorList_Signal2Motors(), this.getSignal2Motor(), null, "signal2Motors", null, 0, -1, Signal2MotorList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signal2SensorListEClass, Signal2SensorList.class, "Signal2SensorList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSignal2SensorList_Signal2Sensors(), this.getSignal2Sensor(), null, "signal2Sensors", null, 0, -1, Signal2SensorList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(motorInitListEClass, MotorInitList.class, "MotorInitList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getMotorInitList_MotorInits(), this.getMotorInit(), null, "motorInits", null, 0, -1, MotorInitList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(motorInitEClass, MotorInit.class, "MotorInit", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getMotorInit_Port(), theSlcoPackage.getPort(), null, "port", null, 1, 1, MotorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMotorInit_Power(), ecorePackage.getEInt(), "power", null, 1, 1, MotorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMotorInit_OutputName(), theNqcPackage.getOutputPortNameEnum(), "outputName", null, 0, 1, MotorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(classListEClass, ClassList.class, "ClassList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getClassList_Classes(), theSlcoPackage.getClass_(), null, "classes", null, 0, -1, ClassList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signal2MotorEClass, Signal2Motor.class, "Signal2Motor", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getSignal2Motor_SignalName(), ecorePackage.getEString(), "signalName", null, 1, 1, Signal2Motor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSignal2Motor_Port(), theSlcoPackage.getPort(), null, "port", null, 1, 1, Signal2Motor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getSignal2Motor_OutputMode(), theNqcPackage.getOutputModeEnum(), "outputMode", null, 1, 1, Signal2Motor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signal2MotorWithDirectionEClass, Signal2MotorWithDirection.class, "Signal2MotorWithDirection", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getSignal2MotorWithDirection_OutputDirection(), theNqcPackage.getDirectionEnum(), "outputDirection", null, 1, 1, Signal2MotorWithDirection.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(signal2SensorEClass, Signal2Sensor.class, "Signal2Sensor", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getSignal2Sensor_SignalName(), ecorePackage.getEString(), "signalName", null, 1, 1, Signal2Sensor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSignal2Sensor_Port(), theSlcoPackage.getPort(), null, "port", null, 1, 1, Signal2Sensor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getSignal2Sensor_SensorIsHigh(), ecorePackage.getEBoolean(), "sensorIsHigh", null, 1, 1, Signal2Sensor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getSignal2Sensor_Treshold(), ecorePackage.getEInt(), "treshold", null, 0, 1, Signal2Sensor.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(slco2NqcEClass, Slco2Nqc.class, "Slco2Nqc", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSlco2Nqc_Signal2MotorList(), this.getSignal2MotorList(), null, "signal2MotorList", null, 1, 1, Slco2Nqc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSlco2Nqc_Signal2SensorList(), this.getSignal2SensorList(), null, "signal2SensorList", null, 1, 1, Slco2Nqc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSlco2Nqc_MotorInitList(), this.getMotorInitList(), null, "motorInitList", null, 1, 1, Slco2Nqc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSlco2Nqc_ClassList(), this.getClassList(), null, "classList", null, 1, 1, Slco2Nqc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSlco2Nqc_SensorInitList(), this.getSensorInitList(), null, "sensorInitList", null, 1, 1, Slco2Nqc.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(sensorInitListEClass, SensorInitList.class, "SensorInitList", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getSensorInitList_SensorInits(), this.getSensorInit(), null, "sensorInits", null, 0, -1, SensorInitList.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(sensorInitEClass, SensorInit.class, "SensorInit", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getSensorInit_SensorName(), theNqcPackage.getSensorNameEnum(), "sensorName", null, 1, 1, SensorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getSensorInit_SensorConfiguration(), theNqcPackage.getSensorConfigEnum(), "sensorConfiguration", null, 1, 1, SensorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getSensorInit_Port(), theSlcoPackage.getPort(), null, "port", null, 1, 1, SensorInit.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		// Create resource
		createResource(eNS_URI);
	}

} //Slco2nqcPackageImpl

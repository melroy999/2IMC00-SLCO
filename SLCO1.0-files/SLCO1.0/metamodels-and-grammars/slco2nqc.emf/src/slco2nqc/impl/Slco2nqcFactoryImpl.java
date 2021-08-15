/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

import slco2nqc.*;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class Slco2nqcFactoryImpl extends EFactoryImpl implements Slco2nqcFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static Slco2nqcFactory init() {
		try {
			Slco2nqcFactory theSlco2nqcFactory = (Slco2nqcFactory)EPackage.Registry.INSTANCE.getEFactory("slco2nqc"); 
			if (theSlco2nqcFactory != null) {
				return theSlco2nqcFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new Slco2nqcFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2nqcFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST: return createSignal2MotorList();
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST: return createSignal2SensorList();
			case Slco2nqcPackage.MOTOR_INIT_LIST: return createMotorInitList();
			case Slco2nqcPackage.MOTOR_INIT: return createMotorInit();
			case Slco2nqcPackage.CLASS_LIST: return createClassList();
			case Slco2nqcPackage.SIGNAL2_MOTOR: return createSignal2Motor();
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION: return createSignal2MotorWithDirection();
			case Slco2nqcPackage.SIGNAL2_SENSOR: return createSignal2Sensor();
			case Slco2nqcPackage.SLCO2_NQC: return createSlco2Nqc();
			case Slco2nqcPackage.SENSOR_INIT_LIST: return createSensorInitList();
			case Slco2nqcPackage.SENSOR_INIT: return createSensorInit();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2MotorList createSignal2MotorList() {
		Signal2MotorListImpl signal2MotorList = new Signal2MotorListImpl();
		return signal2MotorList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2SensorList createSignal2SensorList() {
		Signal2SensorListImpl signal2SensorList = new Signal2SensorListImpl();
		return signal2SensorList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MotorInitList createMotorInitList() {
		MotorInitListImpl motorInitList = new MotorInitListImpl();
		return motorInitList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MotorInit createMotorInit() {
		MotorInitImpl motorInit = new MotorInitImpl();
		return motorInit;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ClassList createClassList() {
		ClassListImpl classList = new ClassListImpl();
		return classList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2Motor createSignal2Motor() {
		Signal2MotorImpl signal2Motor = new Signal2MotorImpl();
		return signal2Motor;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2MotorWithDirection createSignal2MotorWithDirection() {
		Signal2MotorWithDirectionImpl signal2MotorWithDirection = new Signal2MotorWithDirectionImpl();
		return signal2MotorWithDirection;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2Sensor createSignal2Sensor() {
		Signal2SensorImpl signal2Sensor = new Signal2SensorImpl();
		return signal2Sensor;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2Nqc createSlco2Nqc() {
		Slco2NqcImpl slco2Nqc = new Slco2NqcImpl();
		return slco2Nqc;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorInitList createSensorInitList() {
		SensorInitListImpl sensorInitList = new SensorInitListImpl();
		return sensorInitList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorInit createSensorInit() {
		SensorInitImpl sensorInit = new SensorInitImpl();
		return sensorInit;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2nqcPackage getSlco2nqcPackage() {
		return (Slco2nqcPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static Slco2nqcPackage getPackage() {
		return Slco2nqcPackage.eINSTANCE;
	}

} //Slco2nqcFactoryImpl

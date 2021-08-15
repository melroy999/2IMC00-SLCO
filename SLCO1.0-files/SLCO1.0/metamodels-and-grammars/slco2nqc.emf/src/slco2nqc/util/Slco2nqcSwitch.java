/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.util;

import java.util.List;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;

import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.util.Switch;
import slco2nqc.*;

/**
 * <!-- begin-user-doc -->
 * The <b>Switch</b> for the model's inheritance hierarchy.
 * It supports the call {@link #doSwitch(EObject) doSwitch(object)}
 * to invoke the <code>caseXXX</code> method for each class of the model,
 * starting with the actual class of the object
 * and proceeding up the inheritance hierarchy
 * until a non-null result is returned,
 * which is the result of the switch.
 * <!-- end-user-doc -->
 * @see slco2nqc.Slco2nqcPackage
 * @generated
 */
public class Slco2nqcSwitch<T> extends Switch<T> {
	/**
	 * The cached model package
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static Slco2nqcPackage modelPackage;

	/**
	 * Creates an instance of the switch.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2nqcSwitch() {
		if (modelPackage == null) {
			modelPackage = Slco2nqcPackage.eINSTANCE;
		}
	}

	/**
	 * Checks whether this is a switch for the given package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @parameter ePackage the package in question.
	 * @return whether this is a switch for the given package.
	 * @generated
	 */
	@Override
	protected boolean isSwitchFor(EPackage ePackage) {
		return ePackage == modelPackage;
	}

	/**
	 * Calls <code>caseXXX</code> for each class of the model until one returns a non null result; it yields that result.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the first non-null result returned by a <code>caseXXX</code> call.
	 * @generated
	 */
	@Override
	protected T doSwitch(int classifierID, EObject theEObject) {
		switch (classifierID) {
			case Slco2nqcPackage.SIGNAL2_MOTOR_LIST: {
				Signal2MotorList signal2MotorList = (Signal2MotorList)theEObject;
				T result = caseSignal2MotorList(signal2MotorList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SIGNAL2_SENSOR_LIST: {
				Signal2SensorList signal2SensorList = (Signal2SensorList)theEObject;
				T result = caseSignal2SensorList(signal2SensorList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.MOTOR_INIT_LIST: {
				MotorInitList motorInitList = (MotorInitList)theEObject;
				T result = caseMotorInitList(motorInitList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.MOTOR_INIT: {
				MotorInit motorInit = (MotorInit)theEObject;
				T result = caseMotorInit(motorInit);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.CLASS_LIST: {
				ClassList classList = (ClassList)theEObject;
				T result = caseClassList(classList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SIGNAL2_MOTOR: {
				Signal2Motor signal2Motor = (Signal2Motor)theEObject;
				T result = caseSignal2Motor(signal2Motor);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SIGNAL2_MOTOR_WITH_DIRECTION: {
				Signal2MotorWithDirection signal2MotorWithDirection = (Signal2MotorWithDirection)theEObject;
				T result = caseSignal2MotorWithDirection(signal2MotorWithDirection);
				if (result == null) result = caseSignal2Motor(signal2MotorWithDirection);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SIGNAL2_SENSOR: {
				Signal2Sensor signal2Sensor = (Signal2Sensor)theEObject;
				T result = caseSignal2Sensor(signal2Sensor);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SLCO2_NQC: {
				Slco2Nqc slco2Nqc = (Slco2Nqc)theEObject;
				T result = caseSlco2Nqc(slco2Nqc);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SENSOR_INIT_LIST: {
				SensorInitList sensorInitList = (SensorInitList)theEObject;
				T result = caseSensorInitList(sensorInitList);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case Slco2nqcPackage.SENSOR_INIT: {
				SensorInit sensorInit = (SensorInit)theEObject;
				T result = caseSensorInit(sensorInit);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			default: return defaultCase(theEObject);
		}
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Signal2 Motor List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Signal2 Motor List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSignal2MotorList(Signal2MotorList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Signal2 Sensor List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Signal2 Sensor List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSignal2SensorList(Signal2SensorList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Motor Init List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Motor Init List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMotorInitList(MotorInitList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Motor Init</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Motor Init</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMotorInit(MotorInit object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Class List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Class List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseClassList(ClassList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Signal2 Motor</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Signal2 Motor</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSignal2Motor(Signal2Motor object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Signal2 Motor With Direction</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Signal2 Motor With Direction</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSignal2MotorWithDirection(Signal2MotorWithDirection object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Signal2 Sensor</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Signal2 Sensor</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSignal2Sensor(Signal2Sensor object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Slco2 Nqc</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Slco2 Nqc</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSlco2Nqc(Slco2Nqc object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Init List</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Init List</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorInitList(SensorInitList object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Sensor Init</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Sensor Init</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseSensorInit(SensorInit object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch, but this is the last case anyway.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject)
	 * @generated
	 */
	@Override
	public T defaultCase(EObject object) {
		return null;
	}

} //Slco2nqcSwitch

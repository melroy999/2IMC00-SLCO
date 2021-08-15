/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc -->
 * The <b>Factory</b> for the model.
 * It provides a create method for each non-abstract class of the model.
 * <!-- end-user-doc -->
 * @see slco2nqc.Slco2nqcPackage
 * @generated
 */
public interface Slco2nqcFactory extends EFactory {
	/**
	 * The singleton instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	Slco2nqcFactory eINSTANCE = slco2nqc.impl.Slco2nqcFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Signal2 Motor List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Signal2 Motor List</em>'.
	 * @generated
	 */
	Signal2MotorList createSignal2MotorList();

	/**
	 * Returns a new object of class '<em>Signal2 Sensor List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Signal2 Sensor List</em>'.
	 * @generated
	 */
	Signal2SensorList createSignal2SensorList();

	/**
	 * Returns a new object of class '<em>Motor Init List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Motor Init List</em>'.
	 * @generated
	 */
	MotorInitList createMotorInitList();

	/**
	 * Returns a new object of class '<em>Motor Init</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Motor Init</em>'.
	 * @generated
	 */
	MotorInit createMotorInit();

	/**
	 * Returns a new object of class '<em>Class List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Class List</em>'.
	 * @generated
	 */
	ClassList createClassList();

	/**
	 * Returns a new object of class '<em>Signal2 Motor</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Signal2 Motor</em>'.
	 * @generated
	 */
	Signal2Motor createSignal2Motor();

	/**
	 * Returns a new object of class '<em>Signal2 Motor With Direction</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Signal2 Motor With Direction</em>'.
	 * @generated
	 */
	Signal2MotorWithDirection createSignal2MotorWithDirection();

	/**
	 * Returns a new object of class '<em>Signal2 Sensor</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Signal2 Sensor</em>'.
	 * @generated
	 */
	Signal2Sensor createSignal2Sensor();

	/**
	 * Returns a new object of class '<em>Slco2 Nqc</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Slco2 Nqc</em>'.
	 * @generated
	 */
	Slco2Nqc createSlco2Nqc();

	/**
	 * Returns a new object of class '<em>Sensor Init List</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Init List</em>'.
	 * @generated
	 */
	SensorInitList createSensorInitList();

	/**
	 * Returns a new object of class '<em>Sensor Init</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Sensor Init</em>'.
	 * @generated
	 */
	SensorInit createSensorInit();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	Slco2nqcPackage getSlco2nqcPackage();

} //Slco2nqcFactory

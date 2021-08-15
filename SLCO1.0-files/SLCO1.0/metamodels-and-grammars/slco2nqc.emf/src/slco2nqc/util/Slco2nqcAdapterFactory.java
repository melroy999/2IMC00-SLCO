/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.util;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

import slco2nqc.*;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see slco2nqc.Slco2nqcPackage
 * @generated
 */
public class Slco2nqcAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static Slco2nqcPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Slco2nqcAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = Slco2nqcPackage.eINSTANCE;
		}
	}

	/**
	 * Returns whether this factory is applicable for the type of the object.
	 * <!-- begin-user-doc -->
	 * This implementation returns <code>true</code> if the object is either the model's package or is an instance object of the model.
	 * <!-- end-user-doc -->
	 * @return whether this factory is applicable for the type of the object.
	 * @generated
	 */
	@Override
	public boolean isFactoryForType(Object object) {
		if (object == modelPackage) {
			return true;
		}
		if (object instanceof EObject) {
			return ((EObject)object).eClass().getEPackage() == modelPackage;
		}
		return false;
	}

	/**
	 * The switch that delegates to the <code>createXXX</code> methods.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Slco2nqcSwitch<Adapter> modelSwitch =
		new Slco2nqcSwitch<Adapter>() {
			@Override
			public Adapter caseSignal2MotorList(Signal2MotorList object) {
				return createSignal2MotorListAdapter();
			}
			@Override
			public Adapter caseSignal2SensorList(Signal2SensorList object) {
				return createSignal2SensorListAdapter();
			}
			@Override
			public Adapter caseMotorInitList(MotorInitList object) {
				return createMotorInitListAdapter();
			}
			@Override
			public Adapter caseMotorInit(MotorInit object) {
				return createMotorInitAdapter();
			}
			@Override
			public Adapter caseClassList(ClassList object) {
				return createClassListAdapter();
			}
			@Override
			public Adapter caseSignal2Motor(Signal2Motor object) {
				return createSignal2MotorAdapter();
			}
			@Override
			public Adapter caseSignal2MotorWithDirection(Signal2MotorWithDirection object) {
				return createSignal2MotorWithDirectionAdapter();
			}
			@Override
			public Adapter caseSignal2Sensor(Signal2Sensor object) {
				return createSignal2SensorAdapter();
			}
			@Override
			public Adapter caseSlco2Nqc(Slco2Nqc object) {
				return createSlco2NqcAdapter();
			}
			@Override
			public Adapter caseSensorInitList(SensorInitList object) {
				return createSensorInitListAdapter();
			}
			@Override
			public Adapter caseSensorInit(SensorInit object) {
				return createSensorInitAdapter();
			}
			@Override
			public Adapter defaultCase(EObject object) {
				return createEObjectAdapter();
			}
		};

	/**
	 * Creates an adapter for the <code>target</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param target the object to adapt.
	 * @return the adapter for the <code>target</code>.
	 * @generated
	 */
	@Override
	public Adapter createAdapter(Notifier target) {
		return modelSwitch.doSwitch((EObject)target);
	}


	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Signal2MotorList <em>Signal2 Motor List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Signal2MotorList
	 * @generated
	 */
	public Adapter createSignal2MotorListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Signal2SensorList <em>Signal2 Sensor List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Signal2SensorList
	 * @generated
	 */
	public Adapter createSignal2SensorListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.MotorInitList <em>Motor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.MotorInitList
	 * @generated
	 */
	public Adapter createMotorInitListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.MotorInit <em>Motor Init</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.MotorInit
	 * @generated
	 */
	public Adapter createMotorInitAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.ClassList <em>Class List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.ClassList
	 * @generated
	 */
	public Adapter createClassListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Signal2Motor <em>Signal2 Motor</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Signal2Motor
	 * @generated
	 */
	public Adapter createSignal2MotorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Signal2MotorWithDirection <em>Signal2 Motor With Direction</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Signal2MotorWithDirection
	 * @generated
	 */
	public Adapter createSignal2MotorWithDirectionAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Signal2Sensor <em>Signal2 Sensor</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Signal2Sensor
	 * @generated
	 */
	public Adapter createSignal2SensorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.Slco2Nqc <em>Slco2 Nqc</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.Slco2Nqc
	 * @generated
	 */
	public Adapter createSlco2NqcAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.SensorInitList <em>Sensor Init List</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.SensorInitList
	 * @generated
	 */
	public Adapter createSensorInitListAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link slco2nqc.SensorInit <em>Sensor Init</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see slco2nqc.SensorInit
	 * @generated
	 */
	public Adapter createSensorInitAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for the default case.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @generated
	 */
	public Adapter createEObjectAdapter() {
		return null;
	}

} //Slco2nqcAdapterFactory

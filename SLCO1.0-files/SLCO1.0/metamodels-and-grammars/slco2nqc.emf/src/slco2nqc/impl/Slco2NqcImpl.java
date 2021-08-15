/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package slco2nqc.impl;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import slco2nqc.ClassList;
import slco2nqc.MotorInitList;
import slco2nqc.SensorInitList;
import slco2nqc.Signal2MotorList;
import slco2nqc.Signal2SensorList;
import slco2nqc.Slco2Nqc;
import slco2nqc.Slco2nqcPackage;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Slco2 Nqc</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link slco2nqc.impl.Slco2NqcImpl#getSignal2MotorList <em>Signal2 Motor List</em>}</li>
 *   <li>{@link slco2nqc.impl.Slco2NqcImpl#getSignal2SensorList <em>Signal2 Sensor List</em>}</li>
 *   <li>{@link slco2nqc.impl.Slco2NqcImpl#getMotorInitList <em>Motor Init List</em>}</li>
 *   <li>{@link slco2nqc.impl.Slco2NqcImpl#getClassList <em>Class List</em>}</li>
 *   <li>{@link slco2nqc.impl.Slco2NqcImpl#getSensorInitList <em>Sensor Init List</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class Slco2NqcImpl extends EObjectImpl implements Slco2Nqc {
	/**
	 * The cached value of the '{@link #getSignal2MotorList() <em>Signal2 Motor List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignal2MotorList()
	 * @generated
	 * @ordered
	 */
	protected Signal2MotorList signal2MotorList;

	/**
	 * The cached value of the '{@link #getSignal2SensorList() <em>Signal2 Sensor List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignal2SensorList()
	 * @generated
	 * @ordered
	 */
	protected Signal2SensorList signal2SensorList;

	/**
	 * The cached value of the '{@link #getMotorInitList() <em>Motor Init List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMotorInitList()
	 * @generated
	 * @ordered
	 */
	protected MotorInitList motorInitList;

	/**
	 * The cached value of the '{@link #getClassList() <em>Class List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getClassList()
	 * @generated
	 * @ordered
	 */
	protected ClassList classList;

	/**
	 * The cached value of the '{@link #getSensorInitList() <em>Sensor Init List</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSensorInitList()
	 * @generated
	 * @ordered
	 */
	protected SensorInitList sensorInitList;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected Slco2NqcImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return Slco2nqcPackage.Literals.SLCO2_NQC;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2MotorList getSignal2MotorList() {
		return signal2MotorList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetSignal2MotorList(Signal2MotorList newSignal2MotorList, NotificationChain msgs) {
		Signal2MotorList oldSignal2MotorList = signal2MotorList;
		signal2MotorList = newSignal2MotorList;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST, oldSignal2MotorList, newSignal2MotorList);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSignal2MotorList(Signal2MotorList newSignal2MotorList) {
		if (newSignal2MotorList != signal2MotorList) {
			NotificationChain msgs = null;
			if (signal2MotorList != null)
				msgs = ((InternalEObject)signal2MotorList).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST, null, msgs);
			if (newSignal2MotorList != null)
				msgs = ((InternalEObject)newSignal2MotorList).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST, null, msgs);
			msgs = basicSetSignal2MotorList(newSignal2MotorList, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST, newSignal2MotorList, newSignal2MotorList));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Signal2SensorList getSignal2SensorList() {
		return signal2SensorList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetSignal2SensorList(Signal2SensorList newSignal2SensorList, NotificationChain msgs) {
		Signal2SensorList oldSignal2SensorList = signal2SensorList;
		signal2SensorList = newSignal2SensorList;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST, oldSignal2SensorList, newSignal2SensorList);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSignal2SensorList(Signal2SensorList newSignal2SensorList) {
		if (newSignal2SensorList != signal2SensorList) {
			NotificationChain msgs = null;
			if (signal2SensorList != null)
				msgs = ((InternalEObject)signal2SensorList).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST, null, msgs);
			if (newSignal2SensorList != null)
				msgs = ((InternalEObject)newSignal2SensorList).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST, null, msgs);
			msgs = basicSetSignal2SensorList(newSignal2SensorList, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST, newSignal2SensorList, newSignal2SensorList));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MotorInitList getMotorInitList() {
		return motorInitList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetMotorInitList(MotorInitList newMotorInitList, NotificationChain msgs) {
		MotorInitList oldMotorInitList = motorInitList;
		motorInitList = newMotorInitList;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST, oldMotorInitList, newMotorInitList);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMotorInitList(MotorInitList newMotorInitList) {
		if (newMotorInitList != motorInitList) {
			NotificationChain msgs = null;
			if (motorInitList != null)
				msgs = ((InternalEObject)motorInitList).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST, null, msgs);
			if (newMotorInitList != null)
				msgs = ((InternalEObject)newMotorInitList).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST, null, msgs);
			msgs = basicSetMotorInitList(newMotorInitList, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST, newMotorInitList, newMotorInitList));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ClassList getClassList() {
		return classList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetClassList(ClassList newClassList, NotificationChain msgs) {
		ClassList oldClassList = classList;
		classList = newClassList;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__CLASS_LIST, oldClassList, newClassList);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setClassList(ClassList newClassList) {
		if (newClassList != classList) {
			NotificationChain msgs = null;
			if (classList != null)
				msgs = ((InternalEObject)classList).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__CLASS_LIST, null, msgs);
			if (newClassList != null)
				msgs = ((InternalEObject)newClassList).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__CLASS_LIST, null, msgs);
			msgs = basicSetClassList(newClassList, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__CLASS_LIST, newClassList, newClassList));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public SensorInitList getSensorInitList() {
		return sensorInitList;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetSensorInitList(SensorInitList newSensorInitList, NotificationChain msgs) {
		SensorInitList oldSensorInitList = sensorInitList;
		sensorInitList = newSensorInitList;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST, oldSensorInitList, newSensorInitList);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSensorInitList(SensorInitList newSensorInitList) {
		if (newSensorInitList != sensorInitList) {
			NotificationChain msgs = null;
			if (sensorInitList != null)
				msgs = ((InternalEObject)sensorInitList).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST, null, msgs);
			if (newSensorInitList != null)
				msgs = ((InternalEObject)newSensorInitList).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST, null, msgs);
			msgs = basicSetSensorInitList(newSensorInitList, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST, newSensorInitList, newSensorInitList));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
				return basicSetSignal2MotorList(null, msgs);
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
				return basicSetSignal2SensorList(null, msgs);
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
				return basicSetMotorInitList(null, msgs);
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
				return basicSetClassList(null, msgs);
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
				return basicSetSensorInitList(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
				return getSignal2MotorList();
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
				return getSignal2SensorList();
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
				return getMotorInitList();
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
				return getClassList();
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
				return getSensorInitList();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
				setSignal2MotorList((Signal2MotorList)newValue);
				return;
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
				setSignal2SensorList((Signal2SensorList)newValue);
				return;
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
				setMotorInitList((MotorInitList)newValue);
				return;
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
				setClassList((ClassList)newValue);
				return;
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
				setSensorInitList((SensorInitList)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
				setSignal2MotorList((Signal2MotorList)null);
				return;
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
				setSignal2SensorList((Signal2SensorList)null);
				return;
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
				setMotorInitList((MotorInitList)null);
				return;
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
				setClassList((ClassList)null);
				return;
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
				setSensorInitList((SensorInitList)null);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_MOTOR_LIST:
				return signal2MotorList != null;
			case Slco2nqcPackage.SLCO2_NQC__SIGNAL2_SENSOR_LIST:
				return signal2SensorList != null;
			case Slco2nqcPackage.SLCO2_NQC__MOTOR_INIT_LIST:
				return motorInitList != null;
			case Slco2nqcPackage.SLCO2_NQC__CLASS_LIST:
				return classList != null;
			case Slco2nqcPackage.SLCO2_NQC__SENSOR_INIT_LIST:
				return sensorInitList != null;
		}
		return super.eIsSet(featureID);
	}

} //Slco2NqcImpl

/**
 * <copyright>
 * </copyright>
 *
 * $Id$
 */
package promela.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EObjectContainmentEList;
import org.eclipse.emf.ecore.util.InternalEList;

import promela.PromelaPackage;
import promela.init;
import promela.mtype;
import promela.never;
import promela.one_decl;
import promela.primitivetype;
import promela.proctype;
import promela.spec;
import promela.trace;
import promela.utype;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>spec</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * <ul>
 *   <li>{@link promela.impl.specImpl#getName <em>Name</em>}</li>
 *   <li>{@link promela.impl.specImpl#getUtypes <em>Utypes</em>}</li>
 *   <li>{@link promela.impl.specImpl#getMtypes <em>Mtypes</em>}</li>
 *   <li>{@link promela.impl.specImpl#getOne_decls <em>One decls</em>}</li>
 *   <li>{@link promela.impl.specImpl#getProctypes <em>Proctypes</em>}</li>
 *   <li>{@link promela.impl.specImpl#getInits <em>Inits</em>}</li>
 *   <li>{@link promela.impl.specImpl#getNevers <em>Nevers</em>}</li>
 *   <li>{@link promela.impl.specImpl#getTraces <em>Traces</em>}</li>
 *   <li>{@link promela.impl.specImpl#getPrimitivetypes <em>Primitivetypes</em>}</li>
 * </ul>
 * </p>
 *
 * @generated
 */
public class specImpl extends EObjectImpl implements spec {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getUtypes() <em>Utypes</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUtypes()
	 * @generated
	 * @ordered
	 */
	protected EList<utype> utypes;

	/**
	 * The cached value of the '{@link #getMtypes() <em>Mtypes</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMtypes()
	 * @generated
	 * @ordered
	 */
	protected EList<mtype> mtypes;

	/**
	 * The cached value of the '{@link #getOne_decls() <em>One decls</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOne_decls()
	 * @generated
	 * @ordered
	 */
	protected EList<one_decl> one_decls;

	/**
	 * The cached value of the '{@link #getProctypes() <em>Proctypes</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProctypes()
	 * @generated
	 * @ordered
	 */
	protected EList<proctype> proctypes;

	/**
	 * The cached value of the '{@link #getInits() <em>Inits</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInits()
	 * @generated
	 * @ordered
	 */
	protected EList<init> inits;

	/**
	 * The cached value of the '{@link #getNevers() <em>Nevers</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getNevers()
	 * @generated
	 * @ordered
	 */
	protected EList<never> nevers;

	/**
	 * The cached value of the '{@link #getTraces() <em>Traces</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTraces()
	 * @generated
	 * @ordered
	 */
	protected EList<trace> traces;

	/**
	 * The cached value of the '{@link #getPrimitivetypes() <em>Primitivetypes</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPrimitivetypes()
	 * @generated
	 * @ordered
	 */
	protected EList<primitivetype> primitivetypes;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected specImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return PromelaPackage.Literals.SPEC;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, PromelaPackage.SPEC__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<utype> getUtypes() {
		if (utypes == null) {
			utypes = new EObjectContainmentEList<utype>(utype.class, this, PromelaPackage.SPEC__UTYPES);
		}
		return utypes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<mtype> getMtypes() {
		if (mtypes == null) {
			mtypes = new EObjectContainmentEList<mtype>(mtype.class, this, PromelaPackage.SPEC__MTYPES);
		}
		return mtypes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<one_decl> getOne_decls() {
		if (one_decls == null) {
			one_decls = new EObjectContainmentEList<one_decl>(one_decl.class, this, PromelaPackage.SPEC__ONE_DECLS);
		}
		return one_decls;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<proctype> getProctypes() {
		if (proctypes == null) {
			proctypes = new EObjectContainmentEList<proctype>(proctype.class, this, PromelaPackage.SPEC__PROCTYPES);
		}
		return proctypes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<init> getInits() {
		if (inits == null) {
			inits = new EObjectContainmentEList<init>(init.class, this, PromelaPackage.SPEC__INITS);
		}
		return inits;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<never> getNevers() {
		if (nevers == null) {
			nevers = new EObjectContainmentEList<never>(never.class, this, PromelaPackage.SPEC__NEVERS);
		}
		return nevers;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<trace> getTraces() {
		if (traces == null) {
			traces = new EObjectContainmentEList<trace>(trace.class, this, PromelaPackage.SPEC__TRACES);
		}
		return traces;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<primitivetype> getPrimitivetypes() {
		if (primitivetypes == null) {
			primitivetypes = new EObjectContainmentEList<primitivetype>(primitivetype.class, this, PromelaPackage.SPEC__PRIMITIVETYPES);
		}
		return primitivetypes;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case PromelaPackage.SPEC__UTYPES:
				return ((InternalEList<?>)getUtypes()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__MTYPES:
				return ((InternalEList<?>)getMtypes()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__ONE_DECLS:
				return ((InternalEList<?>)getOne_decls()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__PROCTYPES:
				return ((InternalEList<?>)getProctypes()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__INITS:
				return ((InternalEList<?>)getInits()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__NEVERS:
				return ((InternalEList<?>)getNevers()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__TRACES:
				return ((InternalEList<?>)getTraces()).basicRemove(otherEnd, msgs);
			case PromelaPackage.SPEC__PRIMITIVETYPES:
				return ((InternalEList<?>)getPrimitivetypes()).basicRemove(otherEnd, msgs);
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
			case PromelaPackage.SPEC__NAME:
				return getName();
			case PromelaPackage.SPEC__UTYPES:
				return getUtypes();
			case PromelaPackage.SPEC__MTYPES:
				return getMtypes();
			case PromelaPackage.SPEC__ONE_DECLS:
				return getOne_decls();
			case PromelaPackage.SPEC__PROCTYPES:
				return getProctypes();
			case PromelaPackage.SPEC__INITS:
				return getInits();
			case PromelaPackage.SPEC__NEVERS:
				return getNevers();
			case PromelaPackage.SPEC__TRACES:
				return getTraces();
			case PromelaPackage.SPEC__PRIMITIVETYPES:
				return getPrimitivetypes();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case PromelaPackage.SPEC__NAME:
				setName((String)newValue);
				return;
			case PromelaPackage.SPEC__UTYPES:
				getUtypes().clear();
				getUtypes().addAll((Collection<? extends utype>)newValue);
				return;
			case PromelaPackage.SPEC__MTYPES:
				getMtypes().clear();
				getMtypes().addAll((Collection<? extends mtype>)newValue);
				return;
			case PromelaPackage.SPEC__ONE_DECLS:
				getOne_decls().clear();
				getOne_decls().addAll((Collection<? extends one_decl>)newValue);
				return;
			case PromelaPackage.SPEC__PROCTYPES:
				getProctypes().clear();
				getProctypes().addAll((Collection<? extends proctype>)newValue);
				return;
			case PromelaPackage.SPEC__INITS:
				getInits().clear();
				getInits().addAll((Collection<? extends init>)newValue);
				return;
			case PromelaPackage.SPEC__NEVERS:
				getNevers().clear();
				getNevers().addAll((Collection<? extends never>)newValue);
				return;
			case PromelaPackage.SPEC__TRACES:
				getTraces().clear();
				getTraces().addAll((Collection<? extends trace>)newValue);
				return;
			case PromelaPackage.SPEC__PRIMITIVETYPES:
				getPrimitivetypes().clear();
				getPrimitivetypes().addAll((Collection<? extends primitivetype>)newValue);
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
			case PromelaPackage.SPEC__NAME:
				setName(NAME_EDEFAULT);
				return;
			case PromelaPackage.SPEC__UTYPES:
				getUtypes().clear();
				return;
			case PromelaPackage.SPEC__MTYPES:
				getMtypes().clear();
				return;
			case PromelaPackage.SPEC__ONE_DECLS:
				getOne_decls().clear();
				return;
			case PromelaPackage.SPEC__PROCTYPES:
				getProctypes().clear();
				return;
			case PromelaPackage.SPEC__INITS:
				getInits().clear();
				return;
			case PromelaPackage.SPEC__NEVERS:
				getNevers().clear();
				return;
			case PromelaPackage.SPEC__TRACES:
				getTraces().clear();
				return;
			case PromelaPackage.SPEC__PRIMITIVETYPES:
				getPrimitivetypes().clear();
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
			case PromelaPackage.SPEC__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case PromelaPackage.SPEC__UTYPES:
				return utypes != null && !utypes.isEmpty();
			case PromelaPackage.SPEC__MTYPES:
				return mtypes != null && !mtypes.isEmpty();
			case PromelaPackage.SPEC__ONE_DECLS:
				return one_decls != null && !one_decls.isEmpty();
			case PromelaPackage.SPEC__PROCTYPES:
				return proctypes != null && !proctypes.isEmpty();
			case PromelaPackage.SPEC__INITS:
				return inits != null && !inits.isEmpty();
			case PromelaPackage.SPEC__NEVERS:
				return nevers != null && !nevers.isEmpty();
			case PromelaPackage.SPEC__TRACES:
				return traces != null && !traces.isEmpty();
			case PromelaPackage.SPEC__PRIMITIVETYPES:
				return primitivetypes != null && !primitivetypes.isEmpty();
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (name: ");
		result.append(name);
		result.append(')');
		return result.toString();
	}

} //specImpl
